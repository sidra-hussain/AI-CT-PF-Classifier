#!/usr/bin/env python
# coding: utf-8

import os
import pydicom
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from dotenv import load_dotenv
from azure.storage.blob import ContainerClient
import io
import pydicom
import numpy as np

##This function does not work, its getting an un
def load_all_dicom_volumes_from_azure(container_url, folder_path="dataset/", credential=None):
    """
    Loads all DICOM volumes from all patient folders inside the given dataset folder in Azure Blob Storage.
    Args:
        container_url (str): Azure container URL (e.g. "https://<account>.blob.core.windows.net/<container>")
        folder_path (str): Path to the dataset folder inside the container (default: "dataset/")
        credential: Optional credential (SAS token, account key, or DefaultAzureCredential)
    Returns:
        dict: {patient_id: (image_stack, spacing)}
    """

    container = ContainerClient.from_container_url(container_url, credential=credential)
    # Find all unique patient folders (assumes structure: dataset/patient_id/file.dcm)
    patient_folders = set()
    for blob in container.list_blobs(name_starts_with=folder_path):
        parts = blob.name.split('/')
        if len(parts) > 2:  # e.g., ['dataset', 'patient123', 'IM0001.dcm']
            patient_folders.add('/'.join(parts[:2]) + '/')
    results = {}
    for patient_folder in sorted(patient_folders):
        dicoms = []
        for blob in container.list_blobs(name_starts_with=patient_folder):
            if blob.name.lower().endswith('.dcm'):
                stream = container.download_blob(blob.name)
                dicom_bytes = stream.readall()
                dicom_file = io.BytesIO(dicom_bytes)
                dicom = pydicom.dcmread(dicom_file)
                dicoms.append(dicom)
        if not dicoms:
            print(f"Warning: No DICOM files found in {patient_folder}")
            continue
        dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]) if 'ImagePositionPatient' in x else int(x.InstanceNumber))
        image_stack = np.stack([d.pixel_array for d in dicoms])
        try:
            spacing = list(map(float, dicoms[0].PixelSpacing))
            slice_thickness = float(dicoms[0].SliceThickness)
            spacing.append(slice_thickness)
        except Exception:
            spacing = [1.0, 1.0, 1.0]
        patient_id = patient_folder.strip('/').split('/')[-1]
        results[patient_id] = (image_stack, spacing)
    return image_stack, spacing

def load_dicom_volume_local_dir(folder_path):
    # Load all DICOM files in the folder
    dicoms = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.dcm'):
            dicom = pydicom.dcmread(os.path.join(folder_path, filename))
            dicoms.append(dicom)

    # Sort slices by ImagePositionPatient or InstanceNumber
    dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]) if 'ImagePositionPatient' in x else int(x.InstanceNumber))

    # Stack slices into 3D array
    image_stack = np.stack([d.pixel_array for d in dicoms])

    # Get spacing info
    try:
        spacing = list(map(float, dicoms[0].PixelSpacing))  # in-plane spacing
        slice_thickness = float(dicoms[0].SliceThickness)
        spacing.append(slice_thickness)
    except:
        spacing = [1.0, 1.0, 1.0]  # fallback if tags missing

    return image_stack, spacing


def resample_volume(volume, original_spacing, new_spacing=[1.0, 1.0, 1.0]):
    original_spacing = np.array(original_spacing[::-1])  # DICOM order: z, y, x
    new_spacing = np.array(new_spacing)
    
    resize_factor = original_spacing / new_spacing
    new_shape = np.round(np.array(volume.shape) * resize_factor).astype(int)

    volume_sitk = sitk.GetImageFromArray(volume)
    volume_sitk.SetSpacing(original_spacing.tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize([int(s) for s in new_shape[::-1]])
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled = resampler.Execute(volume_sitk)
    return sitk.GetArrayFromImage(resampled)

def normalize_ct(volume, clip_min=-1000, clip_max=400):
    volume = np.clip(volume, clip_min, clip_max)
    volume = (volume - clip_min) / (clip_max - clip_min)  # normalize to [0, 1]
    return volume.astype(np.float32)

def normalize_ct(volume, clip_min=-1000, clip_max=400):
    volume = np.clip(volume, clip_min, clip_max)
    volume = (volume - clip_min) / (clip_max - clip_min)  # normalize to [0, 1]
    return volume.astype(np.float32)


##Create a version of this function that does this from a local directory
def load_and_process_dicom(folder_path):
    # Load environment variables from .env file
    load_dotenv()

    # Get Azure Storage Key from environment
    AZURE_STORAGE_SAS_TOKEN = os.environ.get('AZURE_STORAGE_SAS_TOKEN')

    volume, spacing = load_all_dicom_volumes_from_azure(container_url="https://staictpfscans001.blob.core.windows.net/pf-ct-scans", folder_path = "dataset/", credential=AZURE_STORAGE_SAS_TOKEN)
    resampled = resample_volume(volume, spacing, [1.0, 1.0, 1.0])
    normalized = normalize_ct(resampled)
    return normalized  # shape: (D, H, W)

def get_10_montage_slices(volume):
    """Divide the volume into 10 sections and sample the center slice from each (sequentially)"""
    depth = volume.shape[0]  # z-dimension (axial slices)
    section_size = depth // 10
    slices = []

    for i in range(10):
        start = i * section_size
        end = (i + 1) * section_size if i < 9 else depth
        center_idx = (start + end) // 2
        slices.append(volume[center_idx])

    montage = np.stack(slices, axis=0)  # shape: (10, H, W), ordered top→bottom
    return montage

def preprocess_slice(slice_2d):
    slice_2d = np.clip(slice_2d, -1000, 400)
    slice_2d = (slice_2d + 1000) / 1400
    slice_2d = resize(slice_2d, (224, 224), mode='reflect', anti_aliasing=True)
    return slice_2d.astype(np.float32)

def create_montage_tensor(volume):
    slices = get_10_montage_slices(volume)
    slices = [preprocess_slice(s) for s in slices]
    montage = np.stack(slices)  # shape: (10, H, W)
    montage = montage[:, np.newaxis, :, :]  # (10, 1, H, W)
    montage = np.transpose(montage, (1, 0, 2, 3))  # (1, 10, H, W)
    tensor = torch.tensor(montage, dtype=torch.float32)  # (1, 10, 224, 224)
    tensor = tensor.unsqueeze(0)  # add batch dim: (B=1, C=1, D=10, H, W)
    return tensor

from azure.storage.blob import ContainerClient
import pandas as pd
import io
import os
from dotenv import load_dotenv

def load_labels_csv_from_azure(container_url, csv_blob_path, credential=None):
    """
    Downloads a CSV file from Azure Blob Storage and loads it into a pandas DataFrame.
    Args:
        container_url (str): Azure container URL (e.g. "https://<account>.blob.core.windows.net/<container>")
        csv_blob_path (str): Path to the CSV blob inside the container (e.g. "dataset/labels_binary.csv")
        credential: Optional credential (SAS token, account key, or DefaultAzureCredential)
    Returns:
        pd.DataFrame: DataFrame containing the CSV data
    """
    container = ContainerClient.from_container_url(container_url, credential=credential)
    blob_client = container.get_blob_client(csv_blob_path)
    stream = blob_client.download_blob()
    csv_bytes = stream.readall()
    csv_file = io.BytesIO(csv_bytes)
    df = pd.read_csv(csv_file)
    return df

#figure out how to get this to switch between local and azure, programatically
class DicomMontageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        load_dotenv()
        self.labels_df = load_labels_csv_from_azure(
            container_url="https://staictpfscans001.blob.core.windows.net/pf-ct-scans",
            csv_blob_path=csv_file,
            credential=os.environ.get('AZURE_STORAGE_SAS_TOKEN'))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        patient_id = self.labels_df.iloc[idx]['patient_id']
        label = self.labels_df.iloc[idx]['label']
        dicom_folder = os.path.join(self.root_dir, patient_id)
        
        # Process volume
        volume = load_and_process_dicom(dicom_folder)
        tensor = create_montage_tensor(volume)

        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor.squeeze(0), torch.tensor(label, dtype=torch.long)

def safe_get_label(dataset, idx):
    """Safely extract label from dataset, skip if corrupted."""
    try:
        _, label = dataset[idx]
        return label
    except Exception as e:
        patient_id = getattr(dataset, 'patient_ids', None)
        pid = patient_id[idx] if patient_id is not None else f"index {idx}"
        print(f"Warning: Skipping patient {pid} due to error: {e}")
        return None

def create_train_test_val(root_dir, csv_file, batch_size=8, num_workers=4, test_size=0.25, val_size=0.5, random_state=42):
    """
    Create train, validation, and test DataLoaders with stratified splits,
    filtering out corrupted samples, partitions the dataset into train:validate:test 75%:12.5%:12.5%
    
    Args:
        root_dir (str): Directory containing DICOM images.
        csv_file (str): Path to CSV file with labels.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoaders.
        test_size (float): Fraction of data reserved for test+val.
        val_size (float): Fraction of temp split reserved for validation.
        random_state (int): Random seed.
    
    Returns:
        dict: {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader
        }
    """
    # Load dataset
    dataset = DicomMontageDataset(csv_file=csv_file, root_dir=root_dir)
    
    # Collect valid indices and labels
    valid_indices, valid_labels = [], []
    print("Extracting labels and filtering corrupted samples...")
    for idx in tqdm(range(len(dataset))):
        label = safe_get_label(dataset, idx)
        if label is not None:
            valid_indices.append(idx)
            valid_labels.append(label)
    
    # First split: Train vs Temp (Val+Test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_rel_idx, temp_rel_idx = next(sss1.split(X=valid_labels, y=valid_labels))
    
    train_idx = [valid_indices[i] for i in train_rel_idx]
    temp_idx = [valid_indices[i] for i in temp_rel_idx]
    
    # Second split: Val vs Test
    temp_labels = [valid_labels[i] for i in temp_rel_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    val_rel_idx, test_rel_idx = next(sss2.split(X=temp_labels, y=temp_labels))
    
    val_idx = [temp_idx[i] for i in val_rel_idx]
    test_idx = [temp_idx[i] for i in test_rel_idx]
    
    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader
    }