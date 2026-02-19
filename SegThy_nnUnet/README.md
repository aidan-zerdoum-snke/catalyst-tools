## SegThy Training Dataset - for nnUNet

This dataset was created using the segthy_rename.py script. 

# Methodology
1. US scan volumes were renamed to respect the nnUNet v2 naming conventions (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)
2. US scan volumes were converted to .nii.gz
3. Label volumes were renamed to respect the nnUNet v2 naming conventions
4. Label volumes were converted to .nii.gz
5. Original SegThy labels (1:Thyroid, 2:CCA left, 3:IJV left, 4:CCA right, 5:IJV right) collapsed (1:Thyroid, 2:CCA, 3:IJV)