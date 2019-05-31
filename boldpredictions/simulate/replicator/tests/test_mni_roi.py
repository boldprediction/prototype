import os
import nibabel as ni
import numpy as np

import cortex
from cortex.mni import compute_mni_transform, transform_to_mni, transform_mni_to_subject

from utils import read_json, mni2vox

os.chdir("/auto/k1/fatma/python-packages/replication/code")

subject = "JGfs"
xfmname= "20110321JG_auto2"
volspace = (31, 100, 100)

# Get subject's cortical mask
mask = cortex.db.get_mask(subject, xfmname, "thick")

# Create random brain data
vol = np.zeros(volspace)
vol[mask] = np.random.random((vol[mask].shape))

# Create pycortex Volume
datavol = cortex.Volume(vol, subject, xfmname, cmap='RdBu_r', vmin=0, vmax=1)
# cortex.quickshow(datavol);

# Transform volume to MNI space
print "Transform volume to MNI space"
# 1. Compute transformation matrix
func_to_mni = compute_mni_transform(subject, xfmname)

# 2. Transform to functional volume to MNI space
func_in_mni = transform_to_mni(datavol, func_to_mni)

# 3. Get data
data = func_in_mni.get_data()

# Get MNI coordinates from JSON file
exp_json = read_json("experiments.json")
expname = "Binder2005"
contrast = "contrast2"
coordinates = exp_json[expname]["contrasts"][contrast]["coordinates"]
xyz_all = []
for coord in coordinates:
    xyz_all.append(coord["xyz"])
print "Experiment: {0} Coordinate space: {1}".format(expname, exp_json[expname]["coordinate_space"])

radius = 5
dim = data.shape
roi_mni = np.zeros(dim)

for nxyz, xyz in enumerate(xyz_all):

    # Convert from MNI coordinate to voxel position
    print "Convert MNI mm-space to voxel-coordinate "
    print "MNI Coordinate (mm) {0}/{1}: {2}".format(nxyz, len(xyz_all), xyz)
    xfm_mni = func_in_mni.get_affine()
    xyz = mni2vox(xyz, xfm_mni)
    print "MNI Coordinate (coord) {0}/{1}: {2}".format(nxyz, len(xyz_all), xyz)

    # Create ROI mask in MNI space using the coordinates
    print "Create ROI mask in MNI space"
    MX, MY, MZ = np.ogrid[0:dim[0], 0:dim[1], 0:dim[2]]
    roi_mni_tmp = np.sqrt((MX-xyz[0])**2 + (MY-xyz[1])**2 + (MZ-xyz[2])**2) < radius
    roi_mni[roi_mni_tmp] = 1

print roi_mni.shape

# Transform ROI to Subject space
print "Transform ROI to subject space"
roi_func = transform_mni_to_subject(subject, xfmname, roi_mni, func_to_mni)
roi_func = roi_func.get_data()
# roi_func = np.where(roi_func, 1, 0)  # Make sure things are binary
print roi_func.shape


# Save ROI (MNI space) as NIFTI file
roi_mni_nii = ni.Nifti1Image(roi_mni, xfm_mni)
filename = "/tmp/roi_mni.nii"
roi_mni_nii.to_filename(filename)

# Save ROI (subject space) as NIFTI file
ref_filename = cortex.db.get_xfm(subject, xfmname).reference.get_filename()
xfm_subject = ni.load(ref_filename).get_affine()
roi_func_nii = ni.Nifti1Image(roi_func, xfm_subject)
filename = "/tmp/roi_func.nii"
roi_func_nii.to_filename(filename)

# Create static viewer
roivol = cortex.Volume(roi_func.T, subject, xfmname, cmap='gray_r', mask=mask)
mastervol = dict()
mastervol["roivol"]=roivol
ds = cortex.Dataset(**mastervol)
viewername = "/tmp/roi_viewer"
cortex.webgl.make_static(viewername, ds)
