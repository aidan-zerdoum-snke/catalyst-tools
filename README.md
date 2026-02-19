## catalyst-tools

This repository contains tools provided/developed to support use of the nnUNet V2 framework as part of the TRISH Catalyst grant CAT17.

---

## Project Setup

To setup the nnUNet framework itself, follow the documentation here: https://github.com/MIC-DKFZ/nnUNet

If you meet the requirements there you will be able to use these tools.

## Contents

1. Useful Scripts

2. Analysis Notebooks

3. Slicer Tools

5. SegThy Dataset

## Useful Scripts

segthy_rename.py - used to convert + rename = collapse labels on the original SegThy dataset

evaluate_medpy.py

ijv_cross_sections.py

scanslice_colormap_importer.py

## Analysis Notebooks

CSA_analysis.ipynb - notebook with plotting scripts to make horizontal box-whisker plots from CSA outputs.

## Slicer Tools

SlicerNNUNet.py - contains modified version of the current SlicerNNunet.py extension with added calculators focused around analysis of IJV segments.
