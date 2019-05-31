import os
import tempfile
import numpy as np
import cortex
from repbase import Repbase
from result import Result
from save_3d_views import save_3d_views
# from utils import create_html_table
from experiment import Coordinates
# import time

from matplotlib.pyplot import close as pltclose
import nibabel as ni
from utils import create_html_table, FSLDIR, mni2vox
default_template = os.path.join(FSLDIR, "data", "standard", "MNI152_T1_1mm_brain.nii.gz")

from subject import ContrastData as ContrastDataS


class SubjectAnalysis(Repbase):
    def __call__(self, contrast_data):
        pass


class EmptyAnalysis(SubjectAnalysis):
    def __init__(self):
        pass

    def __call__(self, contrast_data):
        html = "completed"
        return self.make_output(Result(html, contrast_data))


class TotalEffectSize(SubjectAnalysis):
    def __init__(self):
        pass

    def __call__(self, contrast_data):
        total_effect_size = np.sqrt((contrast_data.data ** 2).mean())
        html = "Total effect size (RMS): %0.3f" % total_effect_size
        return self.make_output(Result(html, contrast_data))


class Flatmap(SubjectAnalysis):
    def __init__(self, output_root, path, show_total_effect_size=True, **quickflat_args):
        self.output_root = output_root
        self.path = path

        self.quickflat_args = quickflat_args
        self.show_total_effect_size = show_total_effect_size

    def __call__(self, contrast_data):
        # Create random filename
        # filename = tempfile.mktemp(suffix='.png', dir=self.path, prefix='flatmap-')

        if hasattr(contrast_data,'ref_to_subject'):
            prefix = contrast_data.ref_to_subject.name
        else:
            prefix = 'group'

        if np.sum(contrast_data.data<0) == 0:
            prefix+= '_pmap'

            # tmp_volume = cortex.Volume(contrast_data.data, contrast_data.subject,
            #                                              contrast_data.xfmname,
            #                                             vmin = contrast_data.vmin,
            #                                             vmax = contrast_data.vmax,
            #                                             cmap = contrast_data.vmax,
            #                                             ** self.quickflat_args)
            tmp_volume = contrast_data
            # tmp_volume.data[contrast_data.ref_to_subject._voxels_predicted==False] = np.nan
        else:
            tmp_volume = cortex.Volume(contrast_data.data, contrast_data.subject,
                                                         contrast_data.xfmname,
                                                        ** self.quickflat_args)
            # tmp_volume.data[contrast_data.ref_to_subject._voxels_predicted==False] = np.nan
            tmp_volume.data[tmp_volume.data==0] = np.nan


        filename = '{0}_{1}_{2}_simpleFlatmap.png'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        filename = os.path.join(self.path,filename)
        print filename



        # Save flatmap
        try:
            cortex.quickflat.make_png(os.path.join(self.output_root, filename),
                                     tmp_volume, with_colorbar = False,
                                    with_curvature=True,
                                    cvmin = -2, cvmax = 2,
                                    **self.quickflat_args)
        except:
            cortex.quickflat.make_png(os.path.join(self.output_root, filename),
                                     tmp_volume, with_colorbar = False,
                                    with_curvature=True,
                                    cvmin = -2, cvmax = 2,
                                   recache = True,
                                    **self.quickflat_args)

        # Create HTML pointing to flatmap
        html = "<img src='{filename}'/>".format(d=contrast_data, filename=filename)

        return self.make_output(Result(html, contrast_data))


class ThreeD(SubjectAnalysis):
    def __init__(self, output_root, path, pmap = False, views = range(6),  **volume_args):
        self.output_root = output_root
        self.path = path
        self.views = views
        self.volume_args = volume_args
        self.pmap = pmap

    def __call__(self, contrast_data):
        # Create filename base (endings will be appended to this)
        if hasattr(contrast_data,'ref_to_subject'):
            prefix = contrast_data.ref_to_subject.name
        else:
            prefix = 'group'

        if self.pmap:
            basepath = '{0}_{1}_{2}_3Dpmap'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        else:
            basepath = '{0}_{1}_{2}_3D'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        basepath = os.path.join(self.path,basepath)

        # Save images, this returns their filenames

        if self.pmap:
            tmp_volume = cortex.Volume(contrast_data.thresholded_contrast_05.data, contrast_data.subject,
                                                        contrast_data.xfmname,  ** self.volume_args)
            tmp_volume.vmax = 1.25
            tmp_volume.vmin = 0
            tmp_volume.cmap = "Blues"
            tmp_volume.data[contrast_data.data==0] = np.nan
        else:
            tmp_volume = cortex.Volume(contrast_data.data, contrast_data.subject,
                                                         contrast_data.xfmname,
                                                        ** self.volume_args)
            # tmp_volume.data[contrast_data.ref_to_subject._voxels_predicted==False] = np.nan
            tmp_volume.data[tmp_volume.data==0] = np.nan

        if hasattr(contrast_data,'isPmap'):
            if contrast_data.isPmap:
                tmp_volume.vmin = 0
                tmp_volume.vmax = 1
                tmp_volume.cmap = "Blues"

        # Create HTML pointing to images
        view_order  = ["lateral", "front", "back", "top", "bottom", "medial"]
        view_order = [view_order[v] for v in self.views]

        filenames = save_3d_views(tmp_volume, self.output_root, basepath, view_names=view_order)

        view_template = "<img src='{filename}' class = 'imageclass_{view}'/>"
        surface_template = "<div class = 'surface-{surface}'>\n"
        
        html = "<div class = 'surface-row'>\n"
        for surf, files in filenames.items():
            html+= surface_template.format(surface=surf)
            html+=("\n".join([view_template.format(filename=files[v], view=v) for v in view_order]))
            html+="\n</div> \n"
        html+="</div> \n"

        return self.make_output(Result(html, contrast_data))


class WebGLStatic(SubjectAnalysis):
    def __init__(self, output_root, path):
        self.output_root = output_root
        self.path = path

    def __call__(self, contrast_data):
        # Create static output base
        dirname = tempfile.mktemp(suffix='', dir=self.path, prefix='webgl-')

        # Save static viewer
        cortex.webgl.make_static(os.path.join(self.output_root, dirname), contrast_data)

        # Create HTML with link to viewer
        html_template = "<a href='{filename}'>Static WebGL view</a>"
        html = html_template.format(filename=dirname)

        return self.make_output(Result(html, contrast_data))


class CoordinateAnalysis(SubjectAnalysis):
    def __init__(self):
        pass
    
    def __call__(self, contrast_data):

        # Collect all the mean ROIs for the subject
        results = [self.check_coordinate(c, contrast_data) for c in contrast_data.contrast.coordinates]
        html = create_html_table(results)

        return self.make_output(Result(html, contrast_data))
    
    def check_coordinate(self, coord, contrast_data):

        # Get the ROI mask in MNI space
        mni_roi_mask = coord.get_mni_roi_mask()
        mni_roi_mask = np.where(mni_roi_mask, 1, 0)
        
        # Transform ROI mask (MNI space) to subject space
        # FIXME: sub_mni_xfm is set already in Subject class (func_to_mni)
        # I should better use that here.
        # subj_mni_xfm = cortex.get_mnixfm(contrast_data.subject, contrast_data.xfm)
        # subj_mni_xfm = contrast_data.func_to_mni
        subj_roi_mask = cortex.mni.transform_mni_to_subject(contrast_data.subject, 
                                                            contrast_data.xfmname,
                                                            mni_roi_mask, 
                                                            contrast_data.func_to_mni)
        subj_roi_mask = subj_roi_mask.get_data()

        # Take average of ROI mask over contrast data
        total = np.dot(contrast_data.volume.ravel().data, subj_roi_mask.ravel())
        roi_sum = subj_roi_mask.sum()
        roi_mean = total / roi_sum

        return {"name": coord.name, "xyz": coord.xyz, "mean": roi_mean}


class CoordinateAnalysisRank(CoordinateAnalysis):
    def __init__(self):
        tmp_mni_mask = cortex.get_cortical_mask("MNI","atlas", "thin")
        self.all_non_zero = np.where(tmp_mni_mask != 0)
        self.mni_dim = tmp_mni_mask.shape
        self.n_mni_voxels = len(self.all_non_zero[0])


    def __call__(self, contrast_data):

        #t = time.time()

        # sample possible MNI coordinates
        permuted_order = np.random.permutation(self.n_mni_voxels)
        random_mean = np.zeros([1,100])
        for i in range(100):
            random_xyz = [self.all_non_zero[0][permuted_order[i]]-self.mni_dim[0]/2,
                          self.all_non_zero[1][permuted_order[i]]-self.mni_dim[1]/2,
                          self.all_non_zero[2][permuted_order[i]]-self.mni_dim[2]/2]
            random_mean[0][i] =  self.check_coordinate(Coordinates(random_xyz),contrast_data)['mean']

        # some subjects have nans in some location because they might not have voxels in that region (?)
        random_mean = np.nan_to_num(random_mean)
        results = [self.check_coordinate_rank(c, contrast_data,random_mean) for c in contrast_data.contrast.coordinates]
        html = create_html_table(results,title = "Rank of ROI")
        #print (time.time() - t), "per contrast" # takes about 130 seconds by subject by contrast if 100 random locations

        return self.make_output(Result(html, contrast_data))

    def check_coordinate_rank(self, coord, contrast_data,random_sample):
        roi_mean = self.check_coordinate(coord,contrast_data)['mean']
        return {"name": coord.name, "xyz": coord.xyz, "mean": np.mean(random_sample<=roi_mean)}
    
    
class PermutationTestFlatmap(SubjectAnalysis):
    def __init__(self, output_root, path, **quickflat_args):
        self.output_root = output_root
        self.path = path
        self.quickflat_args = quickflat_args

    def __call__(self,contrast_data):

        filename = tempfile.mktemp(suffix='.png', dir=self.path, prefix='flatPmap-')

        # Save flatmap
        cortex.quickflat.make_png(os.path.join(self.output_root, filename),
                                  contrast_data.permuted_contrast_pval, **self.quickflat_args)

        # Create HTML pointing to flatmap
        html = "<img src='{filename}'/>".format(d=contrast_data, filename=filename)

        return self.make_output(Result(html, contrast_data))


class SubjectCoordinatePlot(SubjectAnalysis):
    ### FIXME: this function is incredibly messy
    def __init__(self, radius, output_root, path, do6colors = False, **quickflat_args):
        self.radius = radius
        self.output_root = output_root
        self.path = path
        self.do6colors = do6colors
        self.quickflat_args = quickflat_args

    def __call__(self, contrast_data):
        contrast = contrast_data.contrast
        results,masks = zip(*[self.check_coordinate(c, contrast_data) for c in contrast.coordinates])
        all_masks = np.zeros(masks[0].shape,dtype=bool)
        for m in masks:
            all_masks = all_masks | m
        if self.do6colors:
            maskim = self.roi_6_colors(all_masks, contrast_data)
        else:
            maskim = self.roi_mask_v2(all_masks, contrast_data)
        html = create_html_table(results)
        html += "<br/>"
        html += maskim

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

        contrast_mni = contrast.thresholded_contrast_05_mni
        mask_mni = contrast.ref_to_subject.predicted_mask_mni
        tmp = roi.ravel()*mask_mni.ravel()

        # Take average of ROI mask over contrast data
        roi_sum = tmp.sum()*1.0
        if roi_sum>0:
            total = np.dot(contrast_mni.ravel(), roi.ravel())
            # roi_sum = roi.sum()
            roi_mean = total / roi_sum
            # Take max of contrast data within ROI mask
            roi_max = contrast_mni[roi > 0].max()
        else:
            roi_mean = np.nan
            roi_max = np.nan

        return {"name": coord.name, "xyz": coord.xyz,
                "mean": roi_mean, "max": roi_max, "roi_sum":roi_sum}, roi

    def roi_mask_v2(self, roimask, contrast_data):

        # Create random filename
        prefix = contrast_data.ref_to_subject.pycortex_surface
        filename = '{0}_{1}_{2}_flatmap.png'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        filename = os.path.join(self.path,filename)
        print filename
        mask = cortex.db.get_mask(contrast_data.ref_to_subject.pycortex_surface,
                                  contrast_data.ref_to_subject.pycortex_transform)
        # a = b
        roimask = roimask*1.0
        roimask = cortex.mni.transform_mni_to_subject(contrast_data.ref_to_subject.pycortex_surface,
                                                      contrast_data.ref_to_subject.pycortex_transform,
                                                      roimask.T,
                                                      contrast_data.ref_to_subject.func_to_mni,
                                                      template=default_template).get_data().T

        roimask[mask==False] = np.nan
        roivol = cortex.Volume(roimask,
                               contrast_data.ref_to_subject.pycortex_surface,
                               contrast_data.ref_to_subject.pycortex_transform,
                               mask = mask)
        # print np.sum(np.isnan(roivol.data))

        # Create flatmap
        contrast_data2 = ContrastDataS(contrast_data.thresholded_contrast_05.data,
                                       contrast_data.ref_to_subject.pycortex_surface,
                                       contrast_data.ref_to_subject.pycortex_transform,
                                       0,#contrast_data.thresholded_contrast_05.vmin,
                                       1.25,#contrast_data.thresholded_contrast_05.vmax,
                                       contrast_data.contrast,
                                       contrast_data.func_to_mni, contrast_data.ref_to_subject,
                                       cmap = "Blues") #YlGn RdPu
        # print ContrastDataS.data.shape
        contrast_data2.data[contrast_data2.ref_to_subject._voxels_predicted ==False ] = np.nan

        fig = cortex.quickflat.make_figure(contrast_data2, with_colorbar = False,with_curvature=True, shadow = 0,
                                           **self.quickflat_args)
                                  # bgcolor='black')

        # Add rois

        #add_hash_layer(fig, roivol, [0,10,255], [0,50000,50001], alpha = 0.5)
        #add_hash_layer(fig, roivol, [255,255,255], [0,8,12])
        add_hash_layer(fig, roivol, [10,0,0], [0,8,8])
        #add_hash_layer(fig, roivol, [10,0,0], [3,8,4])
        #add_hash_layer(fig, roivol, [0,50,0], [4,8,2])
        # add_hash_layer(fig, roivol, [10,10,0], [2,5,2])

        # add_hash_layer(fig, roivol, [0,50,0], [0,5,2])
        # add_hash_layer(fig, roivol, [10,10,0], [2,5,2])
        # add_hash_layer(fig, roivol, [255,255,255], [4,5,2])

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

    def roi_6_colors(self, roimask, contrast_data):

        # Create random filename
        prefix = contrast_data.ref_to_subject.pycortex_surface
        filename = '{0}_{1}_{2}_flatmap_6cols.png'.format(prefix, contrast_data.contrast.experiment.name,
                                                    contrast_data.contrast.contrast_name)
        filename = os.path.join(self.path,filename)
        print filename
        mask = cortex.db.get_mask(contrast_data.ref_to_subject.pycortex_surface,
                                  contrast_data.ref_to_subject.pycortex_transform)
        # a = b
        roimask = roimask*1.0
        roimask = cortex.mni.transform_mni_to_subject(contrast_data.ref_to_subject.pycortex_surface,
                                                      contrast_data.ref_to_subject.pycortex_transform,
                                                      roimask.T,
                                                      contrast_data.ref_to_subject.func_to_mni,
                                                      template=default_template).get_data().T

        roimask[mask==False] = np.nan
        roivol = cortex.Volume(roimask,
                               contrast_data.ref_to_subject.pycortex_surface,
                               contrast_data.ref_to_subject.pycortex_transform,
                               mask = mask)
        # print np.sum(np.isnan(roivol.data))

        # Create flatmap
        contrast_data2 = ContrastDataS(contrast_data.thresholded_contrast_05.data,
                                       contrast_data.ref_to_subject.pycortex_surface,
                                       contrast_data.ref_to_subject.pycortex_transform,
                                       contrast_data.thresholded_contrast_05.vmin,
                                       contrast_data.thresholded_contrast_05.vmax, contrast_data.contrast,
                                       contrast_data.func_to_mni, contrast_data.ref_to_subject)
        # print ContrastDataS.data.shape
        contrast_data2.data = contrast_data2.data.astype('float')
        print contrast_data2.data[0]
        contrast_data2.data[contrast_data2.ref_to_subject._voxels_predicted ==False ] = -1 
        print contrast_data2.data[0]
        tmp =  np.zeros(mask.shape)
        tmp[mask] = contrast_data2.data
        contrast_data2.data = tmp
        print contrast_data2.data[0,0,0]
        roimask[roimask<0.5] = 0
        roimask[roimask>=0.5] = 1

        contrast_data2.data[roimask==1] += 0.5
        tmp = contrast_data2.data[mask]
        
        change_vals = {-1:np.nan, -0.5:-0.5 , 0:np.nan , 0.5:0 ,1: 1.25 , 1.5:0.75}
        #{-1:np.nan, -0.5:-0.5 , 0:np.nan , 0.5:0.25 ,1: 1 , 1.5:0.55}
        
        tmp = np.array([change_vals[i] for i in tmp])
            
        contrast_data2.data = tmp
        contrast_data2.vmin = -0.5
        contrast_data2.vmax = 1.5
        contrast_data2.cmap = 'jet'#'gist_ncar'
        
        fig = cortex.quickflat.make_png(filename, contrast_data2, with_colorbar = False,with_curvature=True)
                                  # bgcolor='black')
        # # Save flatmap
        
        # dpi = 100

        # imsize = fig.get_axes()[0].get_images()[0].get_size()
        # fig.set_size_inches(np.array(imsize)[::-1] / float(dpi))
        # # if bgcolor is None:
        # fig.savefig(filename, transparent=True, dpi=dpi)
        # # else:
        # #     fig.savefig(filename, facecolor=filename, transparent=False, dpi=dpi)
        # fig.clf()
        # pltclose(fig)

        # Create HTML pointing to flatmap
        html = "<img src='{filename}'/>".format(filename=filename)

        return html

def make_hatch_image(dropout_data, height, sampler, size_hash = [0,4,4], recache=False):
    dmap, ee = cortex.quickflat.make(dropout_data, height=height, sampler=sampler, recache=recache)
    hx, hy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))
    hatchspace = 4
    hatchpat = (hx+hy-size_hash[0])%(size_hash[1]*hatchspace) < size_hash[2]
    hatchpat = np.logical_or(hatchpat, hatchpat[:,::-1]).astype(float)
    hatchim = np.dstack([1-hatchpat]*3 + [hatchpat])
    hatchim[:,:,3] *= np.clip(dmap, 0, 1).astype(float)
    hatchim[:,:,3] = hatchim[:,:,3] >0.001
    # print np.sum(np.isnan(hatchpat))
    return hatchim, ee

def add_hash_layer(f, hatch_data, hatch_color, hash_size, height = 1024, recache = False, alpha = 1):
    iy,ix = ((0,-1),(0,-1))
    hatchim, extents = make_hatch_image(hatch_data, height, "nearest", hash_size, recache=recache)
    hatchim[:,:,0] = hatch_color[0]
    hatchim[:,:,1] = hatch_color[1]
    hatchim[:,:,2]= hatch_color[2]
    # if cutout: hatchim[:,:,3]*=co
    dax = f.add_axes((0,0,1,1))
    dax.imshow(hatchim[iy[1]:iy[0]:-1,ix[0]:ix[1]], aspect="equal", interpolation="nearest",
               extent=extents, origin='lower', alpha=alpha)


# class ContrastVolume(cortex.Volume):
#     def __init__(self, data, subject_name, xfm_name, vmin, vmax, contrast, **extra):
#        cortex.Volume.__init__(self,data,
#                               subject_name,
#                               xfm_name,
#                               vmin = vmin,
#                               vmax = vmax,
#                               **extra)
#        self.contrast = contrast
