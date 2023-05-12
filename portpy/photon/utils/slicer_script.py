import sys
import os

print(sys.argv[1])
print(sys.argv[2])
img_dir = sys.argv[1]
# print(patient_folder_path)
# ct = sitk.ReadImage(os.path.join(patient_folder_path, "ct.nrrd"))
# ct_arr = sitk.GetArrayFromImage(ct)
# slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", 'ct')
slicer.util.loadVolume(os.path.join(img_dir, "ct.nrrd"))
ct_node = slicer.util.getNode('ct')
slicer.util.setSliceViewerLayers(background=ct_node)

# slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", 'dose_1d')
slicer.util.loadVolume(os.path.join(img_dir, "dose.nrrd"))
doseNode = slicer.util.getNode('dose')
slicer.util.setSliceViewerLayers(foreground=doseNode)
slicer.util.setSliceViewerLayers(foregroundOpacity=0.4)

# structure_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", 'rt_struct')
slicer.util.loadSegmentation(os.path.join(img_dir, "rtss.seg.nrrd"))
rtss = slicer.util.getNode('rtss')
segmentation = rtss.GetSegmentation()
# n = segmentation.GetNumberOfSegments()
struct_list = sys.argv[2].split(',')
# print(struct_list)
# struct_list = ['PTV', 'CTV', 'BLAD_WALL', 'BLADDER_TRIGONE', 'FEMUR_L', 'FEMUR_R', 'RECT_WALL', 'URETHRA', 'SKIN', 'RIND_0', 'RIND_1', 'RIND_2', 'RIND_3', 'RIND_4']
for i in range(len(struct_list)):
    segment = segmentation.GetNthSegment(i)
    segment.SetName(struct_list[i])

# struct_name node
