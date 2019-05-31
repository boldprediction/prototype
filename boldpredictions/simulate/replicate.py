from celery import Celery

from replicator.utils import read_json
from replicator.experiment import Experiment, ModelHolder
from replicator.subject import Subject
from replicator.subject_analyses import SubjectAnalysis
from replicator.subject_group import SubjectGroup
from replicator.group_analyses import GroupAnalysis
from replicator.result import Result
import cortex
import tempfile
import numpy as np
import h5py
from .celery import Task

import nibabel as ni
import os
from replicator.utils import FSLDIR, mni2vox
default_template = os.path.join(FSLDIR, "data", "standard", "MNI152_T1_1mm_brain.nii.gz")
from matplotlib.pyplot import close as pltclose

from make_roi_figure import make_roi_figure
import json


class WebGL(SubjectAnalysis):
    def __init__(self, dopmap = True,tmp_image_dir = '~/tmp_images'):
        self.dopmap = dopmap
        self.open_browser = False
        self.tmp_image_dir = tmp_image_dir

    def __call__(self, contrast_data):
        # Save static viewer
        contrast_data.vmin = -3
        contrast_data.vmax = 3
        # port = contrast_data.contrast.experiment.extra_info['port']
        # a = cortex.webgl.view(contrast_data,open_browser = self.open_browser, port = port);

        # Create HTML with link to viewer
        #html_template = "<a href='{filename}'>Static WebGL view</a>"
        #html = html_template.format(filename=a)
        # cortex.webgl.make_static('/auto/k1/leila/generative_brain/mysite/simulate/viewers/MNI3',{'blablou1':contrast_data, 'blablou2':cortex.Volume.random('MNI','atlas')})


        contrast_data.data[contrast_data.ref_to_subject.voxels_predicted==0] = np.nan # contrast_data.data==0
        res_dict = {'contrast':contrast_data}

        if self.dopmap:
            res_dict['pmap'] = contrast_data.thresholded_contrast_05
            res_dict['pmap'].data[contrast_data.ref_to_subject.voxels_predicted==0]=np.nan
            res_dict['pmap'].cmap = 'Blues'
            res_dict['pmap'].vmin = 0
            res_dict['pmap'].vmax = 1.25
        # if contrast_data.experiment.extra_info['do_perm']:
        # res_dict['test']= contrast_data.permuted_contrast_pval

        jsonstr = cortex.webgl.make_static_light(self.tmp_image_dir,res_dict)
        

        return self.make_output(Result(jsonstr, jsonstr))


class WebGLGroup(SubjectAnalysis):
    def __init__(self, dopmap = True, tmp_image_dir = '~/tmp_images'): #subjects,
        self.dopmap = dopmap
        self.open_browser = False

        # if self.dopmap:
        #     from replicator.group_analyses import Mean
        #     self.mean_analysis_p = Mean([],smooth=None, pmap = True)
        self.tmp_image_dir = tmp_image_dir
        #self.subjects = subjects

    def __call__(self, contrast):#mean_contrast, subjects_result, contrast):
        # Save static viewer
        # port = contrast_data.contrast.experiment.extra_info['port']
        # a = cortex.webgl.view(contrast_data,open_browser = self.open_browser, port = port);

        # Create HTML with link to viewer
        #html_template = "<a href='{filename}'>Static WebGL view</a>"
        #html = html_template.format(filename=a)
        # cortex.webgl.make_static('/auto/k1/leila/generative_brain/mysite/simulate/viewers/MNI3',{'blablou1':contrast_data, 'blablou2':cortex.Volume.random('MNI','atlas')})

        if not self.dopmap:
            contrast.vmin = -2
            contrast.vmax = 2
            contrast.data[contrast.data==0] = np.nan
            res_dict = {'contrast':contrast}
        else:
            contrast_not_pmap = contrast[0]
            contrast_pmap = contrast[1]
            contrast_not_pmap.vmin = -2
            contrast_not_pmap.vmax = 2
            contrast_not_pmap.data[contrast_not_pmap.data==0] = np.nan
            res_dict = {'contrast':contrast_not_pmap, 'pmap':contrast_pmap}

        #if contrast_data.experiment.extra_info['do_perm']:
        # res_dict['test'] = self.mean_analysis_p.get_group_mean(subjects_result, self.subjects, contrast)[0]

        jsonstr = cortex.webgl.make_static_light(self.tmp_image_dir,res_dict)

        return self.make_output(Result(jsonstr, jsonstr))


class SavePmaps(SubjectAnalysis):

    def __init__(self, tmp_image_dir = '~/tmp_images'): #subjects,

        self.tmp_image_dir = tmp_image_dir

    def __call__(self, contrast):#mean_contrast, subjects_result, contrast):

        #contrast_pmap = contrast[1]
        subj_volume_pmap = contrast[3]
        #allpmaps = np.zeros([len(subj_volume_pmap)+contrast_pmap.shape)])
        tmp = [s.data for s in subj_volume_pmap]
        stacked_tmp = np.stack(tmp)
        stacked_tmp[np.isnan(stacked_tmp)] = 0
        stacked_tmp = stacked_tmp.astype(bool)

        filename = tempfile.mktemp(suffix='.h5f', dir=self.tmp_image_dir, prefix='pmaps-')
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("pmaps",  data=stacked_tmp)

        return self.make_output(Result(filename, filename))



class Replicate(Task):

    def __init__(self):
        import os
        # import shutil
        from replicator.utils import load_config

        self.json_dir = 'replicator/jsons'

        config = load_config(self.json_dir)
        build_dir = config["build_dir"]
        model_dir = config["model_dir"]
        self.model_holder = ModelHolder(self.json_dir)
        tmp_image_dir = './static/simulate/'

        style = 'resources/css/style.css'

        static_dir = "static"
        main_file = os.path.join(build_dir, "index.html")

        if not os.path.exists(build_dir):
            os.mkdir(build_dir)

        if not os.path.exists(os.path.join(build_dir, static_dir)):
            os.mkdir(os.path.join(build_dir, static_dir))

        # from replicator.subject_analyses import TotalEffectSize
        subject_analyses = [#Flatmap(build_dir, static_dir, with_labels=False,
                            #        with_rois=True),
                            #PermutationTestFlatmap(build_dir, static_dir, with_labels=False,with_rois=True),
                            WebGL(tmp_image_dir = tmp_image_dir, dopmap = False)
                            #TotalEffectSize(),
                            #CoordinateAnalysis()
                            #CoordinateAnalysisRank()
                            #ThreeD(build_dir, static_dir),
                            # WebGLStatic(build_dir, static_dir)
                            ]

        subject_analyses_with_p = [#Flatmap(build_dir, static_dir, with_labels=False,
                            #        with_rois=True),
                            #PermutationTestFlatmap(build_dir, static_dir, with_labels=False,with_rois=True),
                            WebGL(tmp_image_dir = tmp_image_dir, dopmap = True),
                            #TotalEffectSize(),
                            #CoordinateAnalysis()
                            #CoordinateAnalysisRank()
                            #ThreeD(build_dir, static_dir),
                            # WebGLStatic(build_dir, static_dir)
                            ]


        subjects_info = read_json('subjects.json', self.json_dir)
        print "subjects_info = "
        print subjects_info
        subjects = [Subject(name=key, model_type="english1000",
                                model_dir=model_dir,
                                analyses=[subject_analyses, subject_analyses_with_p], **subjects_info[key])
                        for key in sorted(subjects_info.keys())]


        from replicator.group_analyses import Mean, Mean_two, GroupCoordinateAnalysis, GroupCoordinateRank
        # group_analyses = [Mean(Flatmap(with_labels=False))]
        group_analyses = [Mean([WebGLGroup(tmp_image_dir = tmp_image_dir, dopmap=False)], #WebGLGroup(subjects)
                               smooth=None, pmap = False),
                          # Mean([WebGLGroup()],
                          #      smooth=None, pmap = True),
                          #Mean([Flatmap(build_dir, static_dir, with_labels=False, with_rois=False),
                          #      GroupCoordinateAnalysis(15, build_dir, static_dir)],
                          #     smooth=None, pmap = True),
                          #GroupCoordinateRank()
                          #ThreeD(build_dir, static_dir)
                          #WebGLStatic(build_dir, static_dir)
                          ]

        group_analyses_with_p = [Mean_two([
                                           WebGLGroup(tmp_image_dir = tmp_image_dir, dopmap = True),
                                           SavePmaps(tmp_image_dir = tmp_image_dir)
                                           ], #WebGLGroup(subjects)
                               smooth=None, pmap = True),
                          # Mean([WebGLGroup()],
                          #      smooth=None, pmap = True),
                          #Mean([Flatmap(build_dir, static_dir, with_labels=False, with_rois=False),
                          #      GroupCoordinateAnalysis(15, build_dir, static_dir)],
                          #     smooth=None, pmap = True),
                          #GroupCoordinateRank()
                          #ThreeD(build_dir, static_dir)
                          #WebGLStatic(build_dir, static_dir)
                          ]

        self.subject_group = SubjectGroup(subjects, [group_analyses, group_analyses_with_p])
        #self.subject_group_with_p = SubjectGroup(subjects_with_p, group_analyses_with_p)

        self.nan_mask = np.load('simulate/replicator/MNI_nan_mask.npy') #/Users/lwehbe/demo/mysite/replicator/
        self.tmp_image_dir = tmp_image_dir

        print("\n created new structure\n ")


    def compute(self, info):
        experiment = Experiment(model_holder = self.model_holder, name = 'temp', image_dir = '.',
                                model_type = 'english1000', json_dir = self.json_dir, **info)
        
        res = experiment.run(self.subject_group, do_pmap = info['do_perm'] )
        result = dict()
        result['group'] = res.data[1][1][1][1][1][1]
        for i in range(len(self.subject_group)):
            result['s_{}'.format(i+1)] = res.data[1][1][2+i][1][1]
        if info['do_perm']:
            result['pmaps'] = res.data[1][1][1][1][2][1]
        else:
            result['pmaps'] = ''

        return result

    def run_ROI_table(self, result_file, roi_list,radius,random_roi_file=''): # roi_random_results,
        sub_volumes = np.array(h5py.File(result_file).get('pmaps'))
        mean_volume = np.mean(sub_volumes.astype(float),axis = 0)
        mean_volume[self.nan_mask==0] = np.nan
        for idx,sub in enumerate(self.subject_group):
            sub_volumes[idx][self.subject_group[idx].predicted_mask_mni==0] = np.nan

        results, masks = zip(*[self.check_coordinate(c, mean_volume, radius) for c in roi_list])
        results_all= dict([ ('s{}'.format(idx),[self.check_coordinate(c,v, radius)[0] for c in roi_list])
                            for idx,v in enumerate(sub_volumes)])
        results_all['MNI'] = results
        all_masks = np.zeros(masks[0].shape,dtype=bool)
        for m in masks:
            all_masks = all_masks | m
        maskim = dict([ ('s{}'.format(idx), self.roi_mask_v2(all_masks, v))
                            for idx,v in enumerate(sub_volumes)])
        maskim['MNI'] = self.roi_mask_v2(all_masks, mean_volume)
        maskim = json.dumps(maskim)
        #if len(roi_random_results)==0:
        if len(random_roi_file)==0: # generate distribution
            random_sample = dict()
            for idx,v in enumerate(sub_volumes):
                tmp_mask = (1- np.isnan(v)) ## keep all the non-nans #(1-all_masks)*(1- np.isnan(v))
                random_sample['s{}'.format(idx)] = self.random_sample(tmp_mask,v,radius)
            tmp_mask = (1-all_masks)*(1- np.isnan(mean_volume))
            random_sample['MNI'] = self.random_sample(tmp_mask,mean_volume,radius)
            random_roi_file = tempfile.mktemp(suffix='.h5f', dir=self.tmp_image_dir, prefix='roirandom-')
            with h5py.File(random_roi_file, 'w') as hf:
                for k,v in random_sample.items():
                    hf.create_dataset(k,data=v)
        else: # load existing
            random_sample = dict()
            with h5py.File(random_roi_file, 'r') as hf:
                for k in hf.keys():
                    random_sample[k] = np.array(hf.get(k))
        roi_image_filename = tempfile.mktemp(suffix='.png', dir=self.tmp_image_dir, prefix='roifigure-')
        make_roi_figure(roi_list, results_all, random_sample, roi_image_filename)
        return random_roi_file, roi_image_filename, maskim

    def random_sample(self, mask, contrast, radius, nP = 10):
        shape = mask.shape
        tmp_mask = np.where(mask.ravel())[0]
        n = tmp_mask.shape[0]
        choices = np.zeros((nP,2))
        index = np.random.choice(n, size=nP, replace=True)
        index_mni = np.unravel_index(tmp_mask[index], shape)
        MX, MY, MZ = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        ## assume that voxels are spaced by 1mm
        for idx,(x,y,z) in enumerate(zip(*index_mni)):
            roi = np.sqrt((MX - x) ** 2 +
                       (MY - y) ** 2 +
                       (MZ - z) ** 2) <= radius
            tmp = roi.ravel()*(1 - np.isnan(contrast)).ravel()
            tmp = tmp * mask.ravel() # keep voxels outside reported: no
            choices[idx,0] = tmp.sum()*1.0
            if idx%50==0:
                print idx
            if choices[idx,0]>0:
                total = np.dot(np.nan_to_num(contrast.ravel()), tmp)
                # roi_sum = roi.sum()
                choices[idx,1] = total / choices[idx,0]
        return choices

    def check_coordinate(self, coord, contrast, radius):
        # Load MNI template image
        template = ni.load(default_template)
        mni_dim = template.shape

        # Get MNI affine transformation between mm-space and coord-space
        transformation = template.get_affine()

        # Convert MNI mm space to coordinate (voxel) space
        coordxyz = [coord.x,coord.y,coord.z]
        xyz = mni2vox(coordxyz, transformation)

        # Draw a sphere around the vox_coord using the 'radius'
        MX, MY, MZ = np.ogrid[0:mni_dim[0], 0:mni_dim[1], 0:mni_dim[2]]
        roiT = np.sqrt((MX - xyz[0]) ** 2 +
                       (MY - xyz[1]) ** 2 +
                       (MZ - xyz[2]) ** 2) <= radius
        roi = roiT.T # also needs to be transposed!

        tmp = roi.ravel()*(1 - np.isnan(contrast)).ravel()
        # Take average of ROI mask over contrast data
        roi_sum = tmp.sum()*1.0
        if roi_sum>0:
            total = np.dot(np.nan_to_num(contrast.ravel()), tmp)
            # roi_sum = roi.sum()
            roi_mean = total / roi_sum
            # Take max of contrast data within ROI mask
            roi_max = contrast[roi > 0].max()
        else:
            roi_mean = np.nan
            roi_max = np.nan

        return {"name": coord.name, "xyz": coordxyz,
                "mean": roi_mean, "max": roi_max, "roi_sum":roi_sum}, roi

    def roi_mask_v2(self, roimask, contrast_data):

        filename = tempfile.mktemp(suffix='.png', dir=self.tmp_image_dir, prefix='roi_masks-')
        mask = cortex.db.get_mask('MNI', 'atlas')
        # a = b
        roimask = roimask*1.0
        roimask[mask==False] = np.nan
        roivol = cortex.Volume(roimask, 'MNI', 'atlas', mask = mask)
        # Create flatmap
        fig = cortex.quickflat.make_figure(cortex.Volume(contrast_data,'MNI','atlas', vmin = 0, vmax = 1, cmap = 'Blues'),
                                           with_colorbar = False,
                                           with_curvature=True,
                                           bgcolor = None)
        # Add rois
        #if not contrast_data.isPmap:
        #    add_hash_layer(fig, roivol, [255,255,255], [0,8,12])
        add_hash_layer(fig, roivol, [10,0,0], [3,8,8])
        dpi = 100
        imsize = fig.get_axes()[0].get_images()[0].get_size()
        fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))
        fig.savefig(filename, transparent=True, dpi=dpi)
        fig.clf()
        pltclose(fig)

        return filename

def make_hatch_image(dropout_data, height, sampler, size_hash=[0,4,4], recache=False):
    dmap, ee = cortex.quickflat.make(dropout_data, height=height, sampler=sampler, recache=recache)
    hx, hy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))
    hatchspace = 4
    hatchpat = (hx+hy+size_hash[0])%(size_hash[1]*hatchspace) < size_hash[2]
    hatchpat = np.logical_or(hatchpat, hatchpat[:,::-1]).astype(float)
    hatchim = np.dstack([1-hatchpat]*3 + [hatchpat])
    hatchim[:,:,3] *= np.clip(dmap, 0, 1).astype(float)
    hatchim[:,:,3] = hatchim[:,:,3] >0.001
    # print np.sum(np.isnan(hatchpat))
    return hatchim, ee

def add_hash_layer(f, hatch_data, hatch_color, hash_size, height = 1024, recache = False):
    iy,ix = ((0,-1),(0,-1))
    hatchim, extents = make_hatch_image(hatch_data, height, "nearest", hash_size, recache=recache)
    hatchim[:,:,0] = hatch_color[0]
    hatchim[:,:,1] = hatch_color[1]
    hatchim[:,:,2]= hatch_color[2]
    # if cutout: hatchim[:,:,3]*=co
    dax = f.add_axes((0,0,1,1))
    dax.imshow(hatchim[iy[1]:iy[0]:-1,ix[0]:ix[1]], aspect="equal", interpolation="nearest",
               extent=extents, origin='lower')
