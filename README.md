# AI-CT-PF-Classifier
This is an ML model that predicts if a patient has Pulmonary Fibrosis or not based on their CT Scans of their Lungs using the SE ResNet50 3D model. 

## Description 

This repository contains preprocessing utilities, training notebooks for binary and multiclass models, and an experiments notebook to compare fine-tuning strategies. The pipeline converts DICOM volumes into 10-slice montages, trains 3D SE-ResNet50 models, and evaluates performance using standard classification metrics.

## Data Processing

Data for this project was acquired from the Open Source Imaging Pulmonary Fibrosis Progression Competition (Kaggle). The repository expects a local `dataset/` folder containing one subfolder per patient with DICOM files, plus CSV label files. Download the competition dataset from Kaggle and place the DICOM series into the `dataset/` folder. 

### Generating Label CSVs with `data-labeling.py`

- Use the `data-labeling.py` script to generate the required `labels_binary.csv` and `labels_multi.csv` files from your raw metadata (e.g., `train.csv`).
- Place the generated CSVs inside the `dataset/` folder before uploading to Azure Blob Storage.
- Example usage:

```bash
python data-labeling.py
```

This will create the label files in the correct format for use with the pipeline.

### Upload Data to Azure Blob Storage

1. **Create a Storage Account and Container**
   - In the Azure Portal, create a Storage Account if you don't have one.
   - Create a container (e.g., `pf-ct-scans`).

2. **Upload the Folder Structure**
   - Use the Azure Portal, Azure Storage Explorer, or the Azure CLI to upload your local `dataset/` folder (with all patient subfolders and CSVs) to the container.
   - The structure inside the container should match the example above.

3. **Verify Structure**
   - After upload, you should see `dataset/` as a folder in the container, with patient subfolders and CSV files inside it.

**Tip:** You can use Azure Storage Explorer for easy drag-and-drop uploads and to visually verify your folder structure.

## Azure Blob Storage Connection

This project supports reading DICOM files and label CSVs directly from Azure Blob Storage.

### 1. Setting up Azure Credentials

Create a `.env` file in the project root with the following content:

```
AZURE_STORAGE_SAS_TOKEN="your_sas_token_here"
```

- Replace `your_sas_token_here` with your actual SAS token.
- The SAS token must have at least `Read` and `List` permissions for the container.

### 2. Generating a SAS Token

- In the Azure Portal, go to your Storage Account > Containers > pf-ct-scans.
- Click "Shared access signature" and select `Read` and `List` permissions.
- Set an expiry time and generate the SAS token.
- Copy the SAS token (everything after the `?`) and paste it into your `.env` file as shown above.

### 3. How the Code Uses the Token

- The code loads the SAS token from `.env` using `python-dotenv`.
- It uses the token to authenticate with Azure Blob Storage for reading DICOM files and CSVs.

### 4. Troubleshooting

- If you see `AuthorizationPermissionMismatch`, your SAS token is missing required permissions.
- Make sure your SAS token is valid and not expired.
- Restart your script or notebook after updating the `.env` file.

## Setting Up Azure Blob Storage with the Correct File Structure

To use Azure Blob Storage with this project, your container (e.g., `pf-ct-scans`) should have the following structure:

```
<container-root>/
    dataset/
        patient_001/
            *.dcm
        patient_002/
            *.dcm
        ...
        labels_binary.csv
        labels_multi.csv
```
- Each patient folder (e.g., `patient_001/`) contains that patient's DICOM files.
- The CSV label files (e.g., `labels_binary.csv`) should be placed inside the `dataset/` folder at the same level as the patient folders.

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
