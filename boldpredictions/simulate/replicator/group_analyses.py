import os
import numpy as np
import tempfile
import cortex
# from cortex import db
from cortex.mni import transform_to_mni
import nibabel as ni
import nipy
from nipy.algorithms import kernel_smooth
# import pickle
# from npp import zs
from matplotlib.pyplot import close as pltclose
import cPickle as pickle

from repbase import Repbase
from utils import create_html_table, FSLDIR, mni2vox
from result import Result
from experiment import Coordinates
from subject_analyses import SubjectAnalysis

default_template = os.path.join(FSLDIR, "data", "standard", "MNI152_T1_1mm_brain.nii.gz")

# from utils import load_config
# use_flirt = load_config()['use_flirt']
use_flirt = False


class GroupAnalysis(Repbase):
    pass

class Mean(Repbase):
    def __init__(self, visualizers, smooth=None, pmap = False, thresh = 0.01, do_1pct=False, mask_pred = False,
                 recompute_mask = False):
        self.visualizers = visualizers
        self.smooth = smooth
        self.pmap = pmap
        # self.nlogthresh = -np.log10(thresh)
        self.thresh = thresh
        self.do_1pct = do_1pct
        self.mask_pred = mask_pred
        self.recompute_mask = recompute_mask
        if not self.recompute_mask:
            self.nan_mask = np.load('simulate/replicator/MNI_nan_mask.npy')

    def __call__(self, subjects_result, subjects, contrast):
        mean_volume, sub_volumes = self.get_group_mean(subjects_result, subjects, contrast)
        outputs = []
        for vis in self.visualizers:
            if isinstance(vis, SubjectAnalysis):
                # It's a subject analysis, just give it the mean
                outputs.append(vis(mean_volume))

            elif isinstance(vis, GroupAnalysis):
                # It's a group analysis, give it mean and individual volumes & contrast
                outputs.append(vis(mean_volume, sub_volumes, contrast))  ### FIXEME I REPLACED SUB_VOLUMES WITH SUBJECT RESULTS

            else:
                raise ValueError('Unknown visualization type: %s' % repr(vis))

        return self.make_output(outputs)

    def get_group_mean(self, subjects_result, subjects, contrast):

        subject_volumes = dict()
        if self.mask_pred:
            for s in subjects_result:
                # mask = s.ref_to_subject.pred_score
                # th = np.percentile(mask,80)
                # mask = mask<th
                mask = s.ref_to_subject._voxels_predicted
                mask = cortex.Volume(mask, s.ref_to_subject.pycortex_surface, s.ref_to_subject.pycortex_transform)
                s.data[mask.data==True] = -1
                if self.pmap:
                    s.thresholded_contrast.data[mask.data==True] = -1
                    subject_volumes[s.subject] = s
        if self.pmap:
            if not self.mask_pred:
                subject_volumes = {sub_result.subject:sub_result.thresholded_contrast_05
                                  for sub_result in subjects_result}
            # for v in subject_volumes.values():
            #     # v.data[v.data < self.nlogthresh] = 0.0
            #     # v.data[v.data >= self.nlogthresh] = 1.0
            #     v.data[v.data>0] = FDR(v.data[v.data>0], self.thresh, do_correction = False)[0]

        else:
            subject_volumes = {sub_result.subject:sub_result for sub_result in subjects_result}

        # print '2'
        for s in subject_volumes.values():
            s.data = np.nan_to_num(s.data)
            # print sum(s.data==0)

        if self.recompute_mask:
            #s2 = []
            #for subject in subjects:
            #    tmp = cortex.Volume(subject._voxels_predicted.astype("float"), subject.pycortex_surface,
            #                        subject.pycortex_transform)
            #    s2.append(transform_to_mni(tmp,subject.func_to_mni, use_flirt=use_flirt).get_data().T)
            s2 = [s.predicted_mask_MNI for s in subjects]
            self.nan_mask = np.mean(np.stack(s2),axis = 0)
            self.nan_mask = self.nan_mask >= 3/8.0
            np.save("simulate/replicator/MNI_nan_mask.npy", self.nan_mask)



        subject_mni_volumes = [transform_to_mni(subject_volumes[subject.pycortex_surface],
                                                subject.func_to_mni, use_flirt=use_flirt).get_data().T
                               for subject in subjects]
        # print np.unique(subject_mni_volumes[0])
        if self.smooth is not None:
            print "Smoothing with %f mm kernel.." % self.smooth
            # Do some smoothing! self.smooth is FWHM of smoother
            atlasim = nipy.load_image(default_template)
            smoother = kernel_smooth.LinearFilter(atlasim.coordmap, 
                                                  atlasim.shape,
                                                  self.smooth)

            unsm_subject_mni_volumes = subject_mni_volumes

            subject_mni_volumes = []
            for svol in unsm_subject_mni_volumes:
                # Create nipy-style Image from volume
                svol_im = nipy.core.image.Image(svol.T, atlasim.coordmap)
                # Pass it through smoother
                sm_svol_im = smoother.smooth(svol_im)
                # Store the result
                subject_mni_volumes.append(sm_svol_im.get_data().T)


        # print '3'
        # for s in subject_mni_volumes:
        #     print np.sum(s.data)
        #     print np.max(s.data)

        # for s in subject_mni_volumes:
        #     print sum(s[:]==0)

        # if not self.pmap:
        group_mean = np.mean(np.stack(subject_mni_volumes),axis =0)
        # else:
        #     group_mean = -np.log10(2*np.mean(np.power(10,-1.0*np.stack(subject_mni_volumes)),axis =0))

        group_mean[self.nan_mask==False] = np.nan
        un_nan = np.isnan(group_mean) * self.nan_mask
        group_mean[un_nan] = 0

        # s2_mean = np.mean(np.stack(s2),axis =0)

        # group_mean[np.abs(group_mean) < 0.1] = np.nan

        if self.pmap or self.do_1pct:
            max_v_volume = 1
        else:
            max_v_volume = 2#np.abs(group_mean).max()/1.5

        if self.do_1pct:
            th = np.percentile(group_mean[group_mean!=0],90)
            group_mean = group_mean>=th

        mean_volume = ContrastVolume(group_mean, 'MNI', 'atlas',
                                    vmin=-max_v_volume,
                                    vmax= max_v_volume,
                                    contrast = contrast,
                                    isPmap = self.pmap)


        sub_volumes = [ContrastVolume(vol, 'MNI', 'atlas',
                                     vmin=-np.abs(vol).max(),
                                     vmax=np.abs(vol).max(),
                                     contrast = contrast)
                       for vol in subject_mni_volumes]

        for idx,v in enumerate(sub_volumes):  # mask the subject volumes for prediction
            v.data[subjects[idx].predicted_mask_mni == False] = np.nan

        return mean_volume, sub_volumes


class Mean_two(Mean):
    def __init__(self, visualizers, smooth=None, pmap = False, thresh = 0.01, do_1pct=False, mask_pred = False,
                 recompute_mask = False):
        Mean.__init__(self, visualizers = visualizers, smooth = smooth, pmap = pmap, thresh = thresh, do_1pct = do_1pct,
		mask_pred = mask_pred, recompute_mask = recompute_mask)

    def __call__(self, subjects_result, subjects, contrast):
        self.pmap = False
        mean_volume, sub_volumes = self.get_group_mean(subjects_result, subjects, contrast)
        self.pmap = True
        mean_volume_pmap, sub_volumes_pmap = self.get_group_mean(subjects_result, subjects, contrast)
        outputs = []
        for vis in self.visualizers:
            if isinstance(vis, SubjectAnalysis):
                # It's a subject analysis, just give it the mean
                outputs.append(vis([mean_volume, mean_volume_pmap, sub_volumes, sub_volumes_pmap, contrast]))

            elif isinstance(vis, GroupAnalysis):
                # It's a group analysis, give it mean and individual volumes & contrast
                outputs.append(vis(mean_volume_pmap, sub_volumes_pmap, contrast))  ### FIXEME I REPLACED SUB_VOLUMES WITH SUBJECT RESULTS

            else:
                raise ValueError('Unknown visualization type: %s' % repr(vis))

        return self.make_output(outputs)


class GroupCoordinateAnalysis(GroupAnalysis):
    def __init__(self, radius, output_root, path):
        self.radius = radius
        self.output_root = output_root
        self.path = path

    def __call__(self, mean_volume, sub_volumes, contrast):
        results = [self.check_coordinate(c, mean_volume) for c in contrast.coordinates]
        html = create_html_table(results)
        html += "<br/>".join([r['maskim'] for r in results])

        return self.make_output(Result(html, contrast))

    def check_coordinate(self, coord, contrast):
        # Load MNI template image
        template = ni.load(default_template)
        mni_dim = template.shape

        # Get MNI affine transformation between mm-space and coord-space
        transformation = template.get_affine()

        # Convert MNI mm space to coordinate (voxel) space
        xyz = mni2vox(coord.xyz, transformation)

        # Draw a sphere around the vox_coord using the 'radius'
        MX, MY, MZ = np.ogrid[0:mni_dim[0], 0:mni_dim[1], 0:mni_dim[2]]
        roiT = np.sqrt((MX - xyz[0]) ** 2 + 
                       (MY - xyz[1]) ** 2 + 
                       (MZ - xyz[2]) ** 2) < self.radius
        roi = roiT.T # also needs to be transposed!

        # Take average of ROI mask over contrast data
        total = np.dot(contrast.data.ravel(), roi.ravel())
        roi_sum = roi.sum()
        roi_mean = total / roi_sum

        # Take max of contrast data within ROI mask
        roi_max = contrast.data[roi > 0].max()

        return {"name": coord.name, "xyz": coord.xyz, 
                "mean": roi_mean, "max": roi_max,
                "maskim": self.debug_roi_mask(roi, contrast)}

    def debug_roi_mask(self, roimask, contrast_data):
        # Create random filename
        filename = tempfile.mktemp(suffix='.png', dir=self.path, prefix='flatmap-')

        roivol = cortex.Volume(roimask, 'MNI', 'atlas')

        # Save flatmap
        f = cortex.quickflat.make_png(os.path.join(self.output_root, filename),
                                  contrast_data, with_colorbar = False,
                                  with_curvature=True,
                                  extra_hatch=(roivol, [0,250,250]),
                                  bgcolor='black')

        # Create HTML pointing to flatmap
        html = "<img src='{filename}'/>".format(filename=filename)

        return html


class GroupCoordinatePlot(GroupAnalysis):
    def __init__(self, radius, output_root, path):
        self.radius = radius
        self.output_root = output_root
        self.path = path

    def __call__(self, mean_volume, sub_volumes, contrast):
        results,masks = zip(*[self.check_coordinate(c, mean_volume) for c in contrast.coordinates])
        ### FIXME: this can me made more efficient
        results_all= dict([ ('s{}'.format(idx),[self.check_coordinate(c,v)[0] for c in contrast.coordinates])
                            for idx,v in enumerate(sub_volumes)])
        results_all['MNI'] = results
        all_masks = np.zeros(masks[0].shape,dtype=bool)
        for m in masks:
            all_masks = all_masks | m
        maskim = self.roi_mask_v2(all_masks, mean_volume)
        coverage = dict()
        for idx,v in enumerate(sub_volumes):
            coverage['s{}'.format(idx)] = self.check_ours_vs_others(v.data,all_masks)
        coverage['MNI'] = self.check_ours_vs_others(mean_volume.data, all_masks)
        random_sample = dict()
        for idx,v in enumerate(sub_volumes):
            tmp_mask = (1-all_masks)*(1- np.isnan(v.data))
            random_sample['s{}'.format(idx)] = self.random_sample(tmp_mask,v)
        tmp_mask = (1-all_masks)*(1- np.isnan(mean_volume.data))
        random_sample['MNI'] = self.random_sample(tmp_mask,mean_volume)
        if mean_volume.isPmap:
            print('This is a P table!')
            filename = '{0}_{1}_table_p.hdf'.format(mean_volume.contrast.experiment.name,
                                                    mean_volume.contrast.contrast_name)
        else:
            print('This is a not a P table!')
            filename = '{0}_{1}_table.hdf'.format(mean_volume.contrast.experiment.name,
                                                    mean_volume.contrast.contrast_name)
        filename = os.path.join(self.path,filename)
        print(filename)
        with open(filename,'wb') as fp:
            pickle.dump(results_all,fp)
            pickle.dump(coverage,fp)
            pickle.dump(random_sample,fp)
        html = create_html_table(results)
        html += "<br/>"
        html += maskim

        return self.make_output(Result(html, contrast))

    def check_ours_vs_others(self,our_v,other_v):
        coverage = dict()
        ours = np.nan_to_num(our_v.ravel())
        others = np.nan_to_num(other_v.ravel())
        prod = ours.dot(others)
        coverage['others_n'] = others.sum()
        coverage['ours_n'] = ours.sum()
        coverage['others'] = prod/coverage['others_n']
        coverage['ours'] = prod/coverage['ours_n']
        return coverage

    def random_sample(self, mask, contrast,nP = 5000):
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
                       (MZ - z) ** 2) < self.radius
            tmp = roi.ravel()*(1 - np.isnan(contrast.data)).ravel()
            tmp = tmp * mask.ravel() # keep voxels outside reported: no
            choices[idx,0] = tmp.sum()*1.0
            if choices[idx,0]>0:
                total = np.dot(np.nan_to_num(contrast.data.ravel()), tmp)
                # roi_sum = roi.sum()
                choices[idx,1] = total / choices[idx,0]
        return choices


    def check_coordinate(self, coord, contrast):
        # Load MNI template image
        template = ni.load(default_template)
        mni_dim = template.shape

        # Get MNI affine transformation between mm-space and coord-space
        transformation = template.get_affine()

        # Convert MNI mm space to coordinate (voxel) space
        xyz = mni2vox(coord.xyz, transformation)

        # Draw a sphere around the vox_coord using the 'radius'
        MX, MY, MZ = np.ogrid[0:mni_dim[0], 0:mni_dim[1], 0:mni_dim[2]]
        roiT = np.sqrt((MX - xyz[0]) ** 2 +
                       (MY - xyz[1]) ** 2 +
                       (MZ - xyz[2]) ** 2) < self.radius
        roi = roiT.T # also needs to be transposed!
        
        tmp = roi.ravel()*(1 - np.isnan(contrast.data)).ravel()
        # Take average of ROI mask over contrast data
        roi_sum = tmp.sum()*1.0
        if roi_sum>0:
            total = np.dot(np.nan_to_num(contrast.data.ravel()), tmp)
            # roi_sum = roi.sum()
            roi_mean = total / roi_sum
            # Take max of contrast data within ROI mask
            roi_max = contrast.data[roi > 0].max()
        else:
            roi_mean = np.nan
            roi_max = np.nan
        
        return {"name": coord.name, "xyz": coord.xyz,
                "mean": roi_mean, "max": roi_max, "roi_sum":roi_sum}, roi

    def debug_roi_mask(self, roimask, contrast_data):
        # Create random filename
        # filename = tempfile.mktemp(suffix='.png', dir=self.path, prefix='flatmap-')
        prefix = 'group'
        if contrast_data.isPmap:
            print('This is a Pmap!')
            filename = '{0}_{1}_{2}_flatPmap.png'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        else:
            print('This is a not a Pmap!')
            filename = '{0}_{1}_{2}_flatmap.png'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        filename = os.path.join(self.path,filename)
        print filename
        # mask = cortex.db.get_mask('MNI', 'atlas')
        roivol = cortex.Volume(roimask, 'MNI', 'atlas')
        # roivol[mask==False] = np.nan

        # Save flatmap
        cortex.quickflat.make_png(os.path.join(self.output_root, filename),
                                  contrast_data,with_colorbar = False,
                                  with_curvature=True,
                                  extra_hatch=(roivol, [255,255,255]),
                                  bgcolor='black')

        # Create HTML pointing to flatmap
        html = "<img src='{filename}'/>".format(filename=filename)

        return html

    def roi_mask_v2(self, roimask, contrast_data):

        # Create random filename
        prefix = 'group'
        if contrast_data.isPmap:
            print('This is a Pmap!')
            filename = '{0}_{1}_{2}_flatPmap.png'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        else:
            print('This is a not a Pmap!')
            filename = '{0}_{1}_{2}_flatmap.png'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        filename = os.path.join(self.path,filename)
        print filename
        mask = cortex.db.get_mask('MNI', 'atlas')
        # a = b
        roimask = roimask*1.0
        roimask[mask==False] = np.nan
        roivol = cortex.Volume(roimask, 'MNI', 'atlas', mask = mask)
        # print np.sum(np.isnan(roivol.data))

        # Create flatmap
        #'YlGn'
        fig = cortex.quickflat.make_figure(contrast_data, with_colorbar = False,
                                  with_curvature=True,
                                  bgcolor='black')

        # Add rois

        if not contrast_data.isPmap:
            add_hash_layer(fig, roivol, [255,255,255], [0,8,12])
        add_hash_layer(fig, roivol, [10,0,0], [3,8,8])
        # add_hash_layer(fig, roivol, [10,10,0], [2,6,2])

        #add_hash_layer(fig, roivol, [0,50,0], [0,6,4])
        #add_hash_layer(fig, roivol, [10,10,0], [2,6,2])
        #add_hash_layer(fig, roivol, [255,255,255], [4,6,2])

        # Save flatmap

        dpi = 100

        imsize = fig.get_axes()[0].get_images()[0].get_size()
        fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))
        # if bgcolor is None:
        fig.savefig(filename, transparent=True, dpi=dpi)
        # else:
        #     fig.savefig(filename, facecolor=filename, transparent=False, dpi=dpi)
        fig.clf()
        pltclose(fig)

        # Create HTML pointing to flatmap
        html = "<img src='{filename}'/>".format(filename=filename)

        return html


def make_hatch_image(dropout_data, height, sampler, size_hash = [0,4,4], recache=False):
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


def create_task(task_names):
    task_img = []
    for task in task_names:
        task_image_file = 'task_images/{0}_pAgF_z_FDR_0.01.nii.gz'.format(task)
        task_img.append(ni.load(task_image_file).get_data().T)
    return np.mean(np.stack(task_img), axis = 0)


class TaskvsContrastPlot(GroupAnalysis):
    def __init__(self, radius, output_root, path):
        self.radius = radius
        self.output_root = output_root
        self.path = path

    def __call__(self, mean_volume, sub_volumes, contrast):

        task_vol = ContrastVolume(create_task(contrast.experiment.tasks),
                                  "MNI", "atlas_2mm", vmin=0, vmax = 15,
                                  contrast = contrast, cmap = "hot")
        results,masks = zip(*[self.check_coordinate(c, task_vol) for c in contrast.coordinates])
        all_masks = np.zeros(masks[0].shape,dtype=bool)
        for m in masks:
            all_masks = all_masks | m
        maskim = self.roi_mask_v2(all_masks, task_vol)
        html = create_html_table(results)
        html += "<br/>"
        html += maskim

        return self.make_output(Result(html, contrast))

    def check_coordinate(self, coord, contrast): # this is 2*2*2mm!!
        # Load MNI template image
        # template = ni.load(default_template)
        template = ni.load('task_images/button_pAgF_z_FDR_0.01.nii.gz')
        mni_dim = template.shape

        # Get MNI affine transformation between mm-space and coord-space
        transformation = template.get_affine()

        # Convert MNI mm space to coordinate (voxel) space
        xyz = mni2vox(coord.xyz, transformation)

        # Draw a sphere around the vox_coord using the 'radius'
        MX, MY, MZ = np.ogrid[0:mni_dim[0], 0:mni_dim[1], 0:mni_dim[2]]
        roiT = np.sqrt((MX - xyz[0]) ** 2 +(MY - xyz[1]) ** 2 + (MZ - xyz[2]) ** 2) < self.radius/2

        roi = roiT.T # also needs to be transposed!

        # Take average of ROI mask over contrast data
        total = np.dot(contrast.data.ravel(), roi.ravel())
        roi_sum = roi.sum()
        roi_mean = total / roi_sum

        # Take max of contrast data within ROI mask
        roi_max = contrast.data[roi > 0].max()

        # # now generate roi again so it's in the 1*1*1 space:
        # MX, MY, MZ = np.ogrid[0:mni_dim[0]*2, 0:mni_dim[1]*2, 0:mni_dim[2]*2]
        # roiT = np.sqrt((MX - xyz[0]) ** 2 +(MY - xyz[1]) ** 2 + (MZ - xyz[2]) ** 2) < self.radius
        #
        # roi = roiT.T # also needs to be transposed!

        return {"name": coord.name, "xyz": coord.xyz,
                "mean": roi_mean, "max": roi_max,
                "roi_sum": roi_sum}, roi


    def roi_mask_v2(self, roimask, contrast_data):

        # Create random filename
        prefix = 'group'
        filename = '{0}_{1}_{2}_task_flatmap.png'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        filename = os.path.join(self.path,filename)
        print filename
        mask_roi = cortex.db.get_mask('MNI', 'atlas_2mm')
        # a = b
        roimask = roimask*1.0
        roimask[mask_roi==False] = np.nan
        roivol = cortex.Volume(roimask, 'MNI', 'atlas_2mm', mask = mask_roi)
        # print np.sum(np.isnan(roivol.data))

        # Create flatmap
        fig = cortex.quickflat.make_figure(contrast_data, with_colorbar = False,
                                  with_curvature=True,
                                  bgcolor='black')

        # Add rois

        #add_hash_layer(fig, roivol, [255,255,255], [0,8,10])
        add_hash_layer(fig, roivol, [10,0,0], [3,8,10])
        # add_hash_layer(fig, roivol, [10,10,0], [2,6,2])

        #add_hash_layer(fig, roivol, [0,50,0], [0,6,4])
        #add_hash_layer(fig, roivol, [10,10,0], [2,6,2])
        #add_hash_layer(fig, roivol, [255,255,255], [4,6,2])

        # Save flatmap

        dpi = 100

        imsize = fig.get_axes()[0].get_images()[0].get_size()
        fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))
        # if bgcolor is None:
        fig.savefig(filename, transparent=True, dpi=dpi)
        # else:
        #     fig.savefig(filename, facecolor=filename, transparent=False, dpi=dpi)
        fig.clf()
        pltclose(fig)

        # Create HTML pointing to flatmap
        html = "<img src='{filename}'/>".format(filename=filename)

        return html



class GroupCoordinateRank(GroupCoordinateAnalysis):
    def __init__(self):
        pass

    def __call__(self, subjects_result, subjects, contrast):
        mean_volume = self.get_group_mean(subjects_result,subjects)
        random_sample = self.check_random_coordinate(mean_volume)
        results = [self.check_coordinate_rank(c, mean_volume,random_sample) for c in contrast.coordinates]
        html = create_html_table(results,title = "Rank of ROI, group analysis")

        return self.make_output(Result(html, contrast))

    def check_coordinate_rank(self,coord, contrast,random_sample):
        real = self.check_coordinate(coord,contrast)['mean']
        return {"name": coord.name, "xyz": coord.xyz, "mean": np.mean(random_sample<=real)}

    def check_random_coordinate(self, contrast):

        all_non_zero = np.where(contrast.data != 0)
        mni_dim = contrast.data.shape
        permuted_order = np.random.permutation(len(all_non_zero[0]))
        random_mean = np.zeros([1,100])
        for i in range(100):
            random_xyz = [all_non_zero[0][permuted_order[i]]-mni_dim[0]/2,all_non_zero[1][permuted_order[i]]-mni_dim[1]/2,all_non_zero[2][permuted_order[i]]-mni_dim[2]/2]
            random_mean[0][i] =  self.check_coordinate(Coordinates(random_xyz),contrast)['mean']

        return random_mean


class GroupPermutationMap(GroupCoordinateAnalysis):
    def __init__(self):
        pass

    def __call__(self, subjects_result, subjects, contrast):
        mean_volume = self.get_group_mean(subjects_result,subjects)
        random_sample = self.check_random_coordinate(mean_volume)
        results = [self.check_coordinate_rank(c, mean_volume,random_sample) for c in contrast.coordinates]
        html = create_html_table(results,title = "Rank of ROI, group analysis")

        return self.make_output(Result(html, contrast))

    def check_coordinate_rank(self,coord, contrast,random_sample):
        real = self.check_coordinate(coord,contrast)['mean']
        return {"name": coord.name, "xyz": coord.xyz, "mean": np.mean(random_sample<=real)}

    def check_random_coordinate(self, contrast):

        all_non_zero = np.where(contrast.data != 0)
        mni_dim = contrast.data.shape
        permuted_order = np.random.permutation(len(all_non_zero[0]))
        random_mean = np.zeros([1,100])
        for i in range(100):
            random_xyz = [all_non_zero[0][permuted_order[i]]-mni_dim[0]/2,all_non_zero[1][permuted_order[i]]-mni_dim[1]/2,all_non_zero[2][permuted_order[i]]-mni_dim[2]/2]
            random_mean[0][i] =  self.check_coordinate(Coordinates(random_xyz),contrast)['mean']

        return random_mean


class ContrastVolume(cortex.Volume):
    def __init__(self, data, subject_name, xfm_name, vmin, vmax, contrast, isPmap=False, **extra):
       cortex.Volume.__init__(self,data,
                              subject_name,
                              xfm_name,
                              vmin = vmin,
                              vmax = vmax,
                              **extra)
       self.contrast = contrast
       self.isPmap = isPmap
       if self.isPmap:
            self.vmin = 0
            self.vmax = 1
            self.cmap = 'Blues'

