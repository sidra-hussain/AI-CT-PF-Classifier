# AI-CT-PF-Classifier
This is an ML model that predicts if a patient has Pulmonary Fibrosis or not based on their CT Scans of their Lungs using the SE ResNet50 3D model. 

## Description 

This repository contains preprocessing utilities, training notebooks for binary and multiclass models, and an experiments notebook to compare fine-tuning strategies. The pipeline converts DICOM volumes into 10-slice montages, trains 3D SE-ResNet50 models, and evaluates performance using standard classification metrics.

## Data 

Data for this project was acquired from the Open Source Imaging Pulmonary Fibrosis Progression Competition (Kaggle). The repository expects a local `dataset/` folder containing one subfolder per patient with DICOM files, plus CSV label files.

labels.csv
0 -> No pulmonary fibrosis 
1 -> pulmonary fibrosis 

## Project files

- `preprocessing.py` — Core preprocessing utilities:
  - load DICOM series, sort slices, and stack into 3D volumes
  - resample volumes to isotropic spacing
  - normalize Hounsfield Units and resize slices
  - extract 10-slice montages and convert to PyTorch tensors
  - `DicomMontageDataset` dataset class and `create_train_test_val` helper that performs stratified train/val/test splitting and returns DataLoaders

- `data-labeling.py` — Utility script used to generate or transform label CSV files from the original dataset metadata (creates `labels_binary.csv` / `labels_multi.csv` used by notebooks).

- `model_finetuning_experiments.ipynb` — Notebook to run systematic fine-tuning experiments. Trains multiple models with different optimizers, learning rates, and freezing strategies and summarizes results for comparison.

- `binaryclass_pf_classifier.ipynb` — Notebook implementing a baseline binary classification training loop (SE-ResNet50 3D), validation, and test evaluation for the two-class problem.

- `multiclass_pf_classifier.ipynb` — Notebook that trains a multi-class version of the model (e.g., normal, mild, moderate, severe) and includes training and evaluation code.

- `preprocessing.ipynb` — Interactive notebook demonstrating the preprocessing steps (loading DICOMs, resampling, normalization, montage creation) and visualizing intermediate results.

- `data-analysis.ipynb` — Exploratory data analysis notebook for inspecting label distributions, metadata, and any dataset quality checks.

- `dataset/` — Directory expected to contain patient DICOM subfolders and CSV label files used by the notebooks.

## Requirements

Install the common Python packages used by the notebooks and preprocessing utilities. Key dependencies include:
- torch (and torchvision appropriate for your CUDA setup)
- monai
- pydicom
- SimpleITK
- scikit-image
- scikit-learn
- pandas
- numpy
- tqdm
- matplotlib
- seaborn

## Quick notes

- Open the relevant notebook in Jupyter/VSCode and run cells sequentially. Use `create_train_test_val(...)` from `preprocessing.py` to obtain DataLoaders for training/validation/testing.
- The experiments notebook (`model_finetuning_experiments.ipynb`) will run multiple training jobs; adjust `num_epochs`, `batch_size`, and `num_workers` before running to fit your hardware.
- For reproducible splits, keep `random_state` fixed when calling `create_train_test_val`.

## Data Usage Disclosure

Anyone may access and use the Competition Data for any purpose, whether commercial or non-commercial, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. 

## References

- Developing approaches to incorporate donor-lung computed tomography images into machine learning models to predict severe primary graft dysfunction after lung transplantation, Ma et al.

## Training the Model

Download the competition dataset from Kaggle and place the DICOM series and provided CSV labels into the `dataset/` folder. Notebooks expect `dataset/labels_binary.csv` or `dataset/labels_multi.csv` depending on the task.
