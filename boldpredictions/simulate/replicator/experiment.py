import nibabel as ni
import numpy as np
import os
import tempfile
import time

from repbase import Repbase
from result import Result
from utils import load_model, mni2vox, FSLDIR
import npp
import pdb
#import PythonMagick


class Experiment(Repbase):

    html_wrap = ("<link rel='stylesheet' type='text/css' href='style.css'>\n<div class='Experiment'>", 
                 "</div>")

    def __init__(self, model_holder='', name='', stimuli=[], contrasts=[], tasks=[], coordinate_space="MNI", DOI='',
                 image_dir='', model_type="english1000", nperm = 1000,
                   **extra_info):
        """
        Initializes experiment structure
        """
        self.name = name
        self.DOI = DOI
        self.extra_info = extra_info
        self.model_type = model_type
        self.nperm = nperm
        self.tasks = tasks
        self.coordinate_space = coordinate_space
        self.model_holder = model_holder
        #self.coords = coordinates
        #self.figures = figures
        self.conditions = self.make_conditions(stimuli)
        self.contrasts = self.make_contrasts(contrasts, image_dir)

    def make_conditions(self, stimuli):
        """
        Creates feature vectors for every condition.
        Basically, returns for each condition a vector in the feature space.
        """
        conditions = dict()
        for condition,word_list in stimuli.items():
            if self.name == "KRNS":
                words = word_list["value"].split(" ")
                words = [w.replace('.','').lower() for w in words]
                words = [w.replace(',','') for w in words]
                conditions[condition] = Condition(words, self)
            else:
                conditions[condition] = Condition(word_list["value"].split(", "), self)
        return conditions

    def make_contrasts(self, contrast_dict, image_dir):
        """
        Make contrast structure. 
        """
        contrasts = dict()
        for contrast, info in sorted(contrast_dict.items()):
            contrasts[contrast] = Contrast(info['condition1'],info['condition2'],
                                           info['coordinates'],
                                           info['figures'],
                                           self,image_dir, contrast_name = contrast)
        return contrasts


    def make_header(self):
        header_template = "<div class='Experiment-info'>Experiment: {e.name}, DOI: <a href=http://doi.org/{e.DOI}>{e.DOI}</a></div>"
        header = header_template.format(e=self)
        return Result(header, None)

    def run(self, subject_group, do_pmap = False):        
        # Make results for each contrast
        results = []
        sorted_contrasts = sorted(self.contrasts.keys())
        for contrast in sorted_contrasts:
            t = time.time()
            print "{0} of experiment {1} ...  ".format(contrast,self.name),
            results.append(self.contrasts[contrast].run(subject_group, do_pmap = do_pmap))
            print "completed in {0} seconds".format(time.time()-t)
        
        # Make output and return
        return self.make_output(results)

    @property
    def model(self):
        """Return word-based (e.g. semantic) feature space, loading it if it hasn't been loaded.
        ATTENTION: THIS ONLY CAN HANDLE ONE MODEL TYPE!!!
        """
        if not hasattr(self, '_model'):
            self._model = self.model_holder.get_model(self.model_type)
        return self._model


class ModelHolder(object):

    def __init__(self, json_dir='./jsons'):
        self.json_dir = json_dir
        self.models = dict()

    def get_model(self,model_type):
        if not model_type in self.models:
            self.models[model_type] = load_model(model_type, self.json_dir)
        return self.models[model_type]



class Condition(object):

    def __init__(self, words,experiment):
        self.words = words
        self.words_in_model = [w for w in words if w in experiment.model.vocab]
        self.nWords = [len(self.words_in_model),len(self.words)]
        if self.nWords[0]>0:
            self.vector = np.vstack([experiment.model[w] for w in self.words_in_model]).mean(0)
        else:
            self.vector = np.zeros([1,experiment.model.ndim]).mean(0) # same size


class Contrast(Repbase):

    def __init__(self, conditions_1, conditions_2, coords, images, experiment,path, contrast_name = 'tmp'):
        self.vector = self.make_contrast_vector([experiment.conditions[c] for c in conditions_1],
                                                [experiment.conditions[c] for c in conditions_2])
        self.coordinates = [Coordinates(**c) for c in coords]
        self.images = images
        self.condition_names = [conditions_1,conditions_2]
        if self.condition_names[1][0] =='baseline' or np.all(experiment.conditions[conditions_2[0]].vector==0):
            self.double_sided = False
        else: self.double_sided = True
        self.experiment = experiment
        self.path = path
        self.contrast_name = contrast_name

    def make_header(self):
        header_template = "<div class='Contrast-info'>Contrast: [{cond_a}] - [{cond_b}]</div>"
        header = header_template.format(cond_a=", ".join(self.condition_names[0]),
                                        cond_b=", ".join(self.condition_names[1]))
        if len(self.images)>0:
            image_template = "<img src='{filename}' class = 'imageclass-paperfigure'>"
            header+="<div class = 'paper-images'>\n"
            for image in self.images:
                #filename = tempfile.mktemp(suffix='.png', dir=self.path, prefix='contrast-')
                #im = PythonMagick.Image(image)
                #im.write(filename)
                header+=image_template.format(filename=image)
            header+="</div> \n"
        return Result(header, None)

    def run(self, subject_group, do_pmap = False):
        return self.make_output(subject_group.run(self, do_pmap = do_pmap))


    def make_contrast_vector(self,cond_1,cond_2):
        # A contrast sometimes comprises a collection of conditions (e.g. check Davis2004).
        # We take the mean over the condition feature vectors for each contrast
        # FIXME: we can use nWords to normalize according to how many words we have in each sub condition
        vector1 = np.vstack([cond.vector for cond in cond_1]).mean(0)
        vector2 = np.vstack([cond.vector for cond in cond_2]).mean(0)
        #vector = np.nan_to_num(npp.zs(vector1)) - np.nan_to_num(npp.zs(vector2))  # Fix: why npp.zs? should we do it per subcondition?
        return vector1 - vector2


    @property
    def permuted_vectors(self):

        if not hasattr(self, '_permuted_vectors'):
            if not self.double_sided:
                # do bootstrap
                words_cond1 = [w for w in [self.experiment.conditions[cond].words_in_model for cond in self.condition_names[0]]]
                words_cond1 = [w for s in words_cond1 for w in s]
                nwords1 = len(words_cond1)
                self._permuted_vectors = np.zeros([self.experiment.nperm, self.experiment.model.ndim])
                for i in range(self.experiment.nperm):
                    pw = np.random.randint(nwords1, size = nwords1)
                    tmpcond1 = Condition([words_cond1[iw] for iw in pw],self.experiment)
                    tmpcond2 = Condition([''],self.experiment)
                    self._permuted_vectors[i,:] = self.make_contrast_vector([tmpcond1],[tmpcond2])
            else:
                # do permutation test
                words_cond1 = [w for w in [self.experiment.conditions[cond].words_in_model for cond in self.condition_names[0]]]
                words_cond1 = [w for s in words_cond1 for w in s]
                nwords1 = len(words_cond1)
                words_cond2 = [w for w in [self.experiment.conditions[cond].words_in_model for cond in self.condition_names[1]]]
                words_cond2 = [w for s in words_cond2 for w in s]
                all_words = words_cond1+words_cond2
                self._permuted_vectors = np.zeros([self.experiment.nperm, self.experiment.model.ndim])
                for i in range(self.experiment.nperm):
                    pw = np.random.permutation(all_words)
                    tmpcond1 = Condition(pw[:nwords1],self.experiment)
                    tmpcond2 = Condition(pw[nwords1:],self.experiment)
                    self._permuted_vectors[i,:] = self.make_contrast_vector([tmpcond1],[tmpcond2])
            print 'generated {0} randomized vectors for contrast {1}'.format(self.experiment.nperm,self.condition_names)
        return self._permuted_vectors



class Coordinates(object):

    def __init__(self, xyz, name=None, zscore=None, size=8, **extra_info):
        self.xyz = xyz
        self.name = name
        self.zscore = zscore
        self.size = size
        self.extra_info = extra_info

    def get_mni_roi_mask(self):

        """Given an MNI coordinate xyz (in mm) returns the ROI mask in MNI space."""
        radius = self.size

        # Load MNI template image
        default_template = os.path.join(FSLDIR, "data", "standard", "MNI152_T1_1mm_brain.nii.gz")
        template = ni.load(default_template)

        # Get MNI affine transformation between mm-space and coord-space
        transformation = template.get_affine()

        # Convert MNI mm space to coordinate (voxel) space
        xyz = mni2vox(self.xyz, transformation)

        # Create an ROI mask
        # 0. Get MNI dims
        mni_dim = template.shape

        # 1. Draw a sphere around the vox_coord using the 'radius'
        MX, MY, MZ = np.ogrid[0:mni_dim[0], 0:mni_dim[1], 0:mni_dim[2]]
        roi = np.sqrt((MX-xyz[0])**2 + (MY-xyz[1])**2 + (MZ-xyz[2])**2) < radius

        return roi
