import os
import tables
import cortex
import numpy as np

from repbase import Repbase
from result import Result
import npp

# from utils import load_config
# use_flirt = load_config()['use_flirt']
use_flirt = False

class Subject(Repbase):

    def __init__(self, name, pycortex_surface, xfm, models, model_type, 
                 model_dir, analyses, **extra_info):
        self.name = name
        self.model_path = os.path.join(model_dir[model_type], models[model_type])
        # These things have to be str and not unicode for pycortex C compatibility
        self.pycortex_surface = str(pycortex_surface)
        self.pycortex_transform = str(xfm)
        print "xfm = "
        print self.pycortex_transform
        print "surface = "
        print self.pycortex_surface
        self.analyses = analyses
        self.extra_info = extra_info
        self.func_to_mni = cortex.db.get_mnixfm(self.pycortex_surface,self.pycortex_transform)

    @property
    def weights(self):
        if not hasattr(self, "_weights"):
            with tables.open_file(self.model_path) as tf:
                self._weights = tf.root.udwt.read()
                self.pred_score = tf.root.corr.read()
                threshold = 0.05
                self._voxels_predicted = self.pred_score>threshold
                self._weights[:,self._voxels_predicted==False] = 0
        return self._weights

    @property
    def voxels_predicted(self):
        if not hasattr(self, "_voxels_predicted"):
            with tables.open_file(self.model_path) as tf:
                self.pred_score = tf.root.corr.read()
                threshold = 0.05
                self._voxels_predicted = self.pred_score>threshold
        return self._voxels_predicted

    @property
    def predicted_mask_mni(self):
        if not hasattr(self, "_predicted_mask_mni"):
            tmp = cortex.Volume(self.voxels_predicted.astype("float"), self.pycortex_surface,self.pycortex_transform)
            self._predicted_mask_mni = cortex.mni.transform_to_mni(tmp,self.func_to_mni,
                                                                     use_flirt=use_flirt).get_data().T
            self._predicted_mask_mni = (self._predicted_mask_mni>0)*1.0
        return self._predicted_mask_mni


    def make_header(self):
        header = "<div class='Subject-info'>Subject: {s.name}</div>".format(s=self)
        return Result(header, None)

    def run(self, contrast, do_pmap = False):
        """
        run all contrasts through model
        :param contrast: Contrast object
        :return: data volume containing the map
        """
        # Create contrast Volume
        data = self.weights.T.dot(contrast.vector)
        # print  data[self._voxels_predicted]
        data[self.voxels_predicted] = npp.rs(data[self.voxels_predicted])
        # print  data[self.voxels_predicted]
        # data[np.isnan(data)]  = 0
        # print  data[self.voxels_predicted]
        # contrast.data[contrast.data==0] = np.nan
        # data = np.nan_to_num(data)
        data[self.voxels_predicted==False] = np.nan
        contrast_data = ContrastData(np.nan_to_num(data),
                                     # data,
                                     self.pycortex_surface,
                                     self.pycortex_transform,
                                     vmin=-np.abs(np.nan_to_num(data)).max(),
                                     vmax= np.abs(np.nan_to_num(data)).max(),
                                     contrast = contrast,
                                     func_to_mni = self.func_to_mni,
                                     ref_to_subject = self)

        # Run analyses
	if isinstance(self.analyses[0], (list)):
            if do_pmap:
                results = [analysis(contrast_data) for analysis in self.analyses[1]]
            else:
                results = [analysis(contrast_data) for analysis in self.analyses[0]]
        else:
            results = [analysis(contrast_data) for analysis in self.analyses]
        return [self.make_output(results),contrast_data]
    

def FDR(vector, q, do_correction = False):
    original_shape = vector.shape
    vector = vector.flatten()
    N = vector.shape[0]
    sorted_vector = sorted(vector)
    if do_correction:
        C = np.sum([1.0/i for i in range(N)])
    else:
        C = 1.0
    thresh = 0
    #a=b
    for i in range(N-1, 0, -1):
        if sorted_vector[i]<= (i*1.0)/N*q/C:
            thresh = sorted_vector[i]
            break
    thresh_vector = vector<=thresh
    thresh_vector = thresh_vector.reshape(original_shape)
    thresh_vector = thresh_vector*1.0
    print "FDR threshold is : {}, {} voxels rejected".format(thresh, thresh_vector.sum())
    return thresh_vector, thresh


class ContrastData(cortex.Volume):
    def __init__(self, data, subject_name, xfm_name, vmin, vmax, contrast, func_to_mni, ref_to_subject,cmap='RdBu_r'):
       cortex.Volume.__init__(self,data,
                              subject_name,
                              xfm_name,
                              vmin = vmin,
                              vmax = vmax,
                              cmap = cmap)
       self.contrast = contrast
       self.func_to_mni = func_to_mni
       self.ref_to_subject = ref_to_subject


    @property
    def permuted_contrast_pval(self):
        if not hasattr(self, "_permuted_contrast"):
            p_contrast_vecs = np.dot(self.contrast.permuted_vectors,self.ref_to_subject.weights)
            contrast_vect = np.dot(self.contrast.vector, self.ref_to_subject.weights)
            if self.contrast.double_sided:
                counts = (contrast_vect<=p_contrast_vecs).mean(0)
                counts[counts==0] = 1.0/p_contrast_vecs.shape[0] # can't have pval=0
                counts[self.ref_to_subject.voxels_predicted == False] = 0 # these are areas with no predictions
                p_map = counts # -np.log10(counts)
            else:
                #stds = p_contrast_vecs.std(0)
                #p_map = np.norm.cdf(np.zeros(contrast_vect.shape),loc = contrast_vect, scale=stds)
                #p_map = -np.log10(p_map)
                counts = (np.zeros_like(contrast_vect)>=p_contrast_vecs).mean(0)
                counts[counts==0] = 1.0/p_contrast_vecs.shape[0] # can't have pval=0
                counts[self.ref_to_subject.voxels_predicted == False] = 0 # these are areas with no predictions
                p_map = counts #-np.log10(counts)
            self._permuted_contrast_pval = cortex.Volume(p_map,
                                                         self.ref_to_subject.pycortex_surface,
                                                         self.ref_to_subject.pycortex_transform,
                                                         vmin=0,vmax=1)
        return self._permuted_contrast_pval


    @property
    def thresholded_contrast_05(self):
        if not hasattr(self, "_thresholded_contrast_05"):
            thresholded_contrast_05 = self.permuted_contrast_pval.data
            thresholded_contrast_05[thresholded_contrast_05>0] = FDR(thresholded_contrast_05[thresholded_contrast_05>0],
                                                                     0.05, do_correction=False)[0]
            self._thresholded_contrast_05 = cortex.Volume(thresholded_contrast_05,
                                                         self.ref_to_subject.pycortex_surface,
                                                         self.ref_to_subject.pycortex_transform,
                                                         vmin=-0.5,vmax=0.5)
        return self._thresholded_contrast_05

    @property
    def thresholded_contrast_05_mni(self):
        if not hasattr(self, "_thresholded_contrast_05_mni"):
            self._thresholded_contrast_05_mni = cortex.mni.transform_to_mni(self.thresholded_contrast_05,
                                                                            self.func_to_mni,
                                                                            use_flirt=use_flirt).get_data().T
        return self._thresholded_contrast_05_mni

    @property
    def thresholded_contrast_01(self):
        if not hasattr(self, "_thresholded_contrast_01"):
            thresholded_contrast_01 = self.permuted_contrast_pval.data
            thresholded_contrast_01[thresholded_contrast_01>0] = FDR(thresholded_contrast_01[thresholded_contrast_01>0],
                                                                     0.01, do_correction=False)[0]
            self._thresholded_contrast_01 = cortex.Volume(thresholded_contrast_01,
                                                         self.ref_to_subject.pycortex_surface,
                                                         self.ref_to_subject.pycortex_transform,
                                                         vmin=-0.5,vmax=0.5)
        return self._thresholded_contrast_01

    @property
    def thresholded_contrast_01_mni(self):
        if not hasattr(self, "_thresholded_contrast_01_mni"):
            self._thresholded_contrast_01_mni = cortex.mni.transform_to_mni(self.thresholded_contrast_01,
                                                                            self.func_to_mni,
                                                                            use_flirt=use_flirt).get_data().T
        return self._thresholded_contrast_01_mni
