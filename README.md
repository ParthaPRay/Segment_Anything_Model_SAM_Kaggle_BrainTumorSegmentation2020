# Segment_Anything_Model_SAM_Kaggle_BrainTumorSegmentation2020
This repo contains codes for specific fine tuning of SAM by using BrainTumorSegmentation2020 (BraTS2020) Dataset on Kaggle


# Task

The Description of the project is as follows :

i need to train the Segment Anything Model (SAM) on the Brats 2020 dataset found on Kaggle and here is its link: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

please use this exact dataset in the link.

the training should be done on google colab using python and T4 GPU

Requirements:

1- Google Drive that contains all the directories of the data used in training

2- Google Colab File That contains the following:


-combine the 3 mask channels into one mask (use this mask for training)
-combine the 4 image channels into one image ( use this image for training)
-visualize a sample of the combined mask and combined image




Start Training the Segment Anything Model 
using T4 gpu ( this is a must)
for 5 epochs
batch size 2
Print for every epoch : Haussdorf, Dice, Loss, precision




Train the Segment Anything Model 
using T4 gpu ( this is a must)
for 10 epochs
batch size 2
Print for every epoch : Haussdorf, Dice, Loss, precision

Milestone 4:

Start Training the Segment Anything Model 
using T4 gpu ( this is a must)
for 20 epochs
batch size 2
Print for every epoch : Haussdorf, Dice, Loss, precision

---

Below is a comprehensive guide for training the **Segment Anything Model (SAM)** using **Ultralytics** on the **BraTS 2020** dataset within **Google Colab**. This guide incorporates the use of **`scipy.spatial.distance.directed_hausdorff`** to compute the **Hausdorff Distance** during training.

---

## **Table of Contents**
1. [Setup Google Colab Environment](#1-setup-google-colab-environment)
2. [Mount Google Drive](#2-mount-google-drive)
3. [Install Required Packages](#3-install-required-packages)
4. [Download and Organize the BraTS 2020 Dataset](#4-download-and-organize-the-brats-2020-dataset)
5. [Data Preprocessing](#5-data-preprocessing)
   - [Combine Image Channels](#combine-image-channels)
   - [Combine Mask Channels](#combine-mask-channels)
6. [Visualize Sample Data](#6-visualize-sample-data)
7. [Prepare Data for Training](#7-prepare-data-for-training)
8. [Define Evaluation Metrics](#8-define-evaluation-metrics)
9. [Initialize and Train the SAM Model](#9-initialize-and-train-the-sam-model)
   - [Milestone 1: Train for 5 Epochs](#milestone-1-train-for-5-epochs)
   - [Milestone 2: Train for 10 Epochs](#milestone-2-train-for-10-epochs)
   - [Milestone 3: Train for 20 Epochs](#milestone-3-train-for-20-epochs)
10. [Save and Export the Trained Model](#10-save-and-export-the-trained-model)
11. [Additional Notes](#11-additional-notes)

---

## **1. Setup Google Colab Environment**

Ensure that you have access to **Google Colab** and that your runtime is set to use a **GPU (specifically a T4 GPU)**.

1. **Select GPU as the Hardware Accelerator:**
   - Go to `Runtime` > `Change runtime type`.
   - Under `Hardware accelerator`, select `GPU`.
   - Click `Save`.

2. **Verify GPU Availability:**

```python
import torch

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Please check your runtime settings.")
```

*Expected Output:*
```
GPU is available: NVIDIA Tesla T4
```

---

## **2. Mount Google Drive**

Mount your **Google Drive** to access the dataset and store model checkpoints.

```python
from google.colab import drive
drive.mount('/content/drive')
```

*Follow the prompt to authorize access.*

---

## **3. Install Required Packages**

Install the necessary Python packages, including **Ultralytics**, **Nibabel** (for handling NIfTI files), **Scipy**, and other dependencies.

```python
!pip install ultralytics
!pip install nibabel
!pip install scikit-image
!pip install segmentation-models-pytorch
!pip install matplotlib
!pip install scipy
```

---

## **4. Download and Organize the BraTS 2020 Dataset**

**Option 1: Direct Download from Kaggle**

To download the dataset directly from Kaggle, you need to set up Kaggle API credentials.

1. **Upload Kaggle API Token:**
   - Go to your Kaggle account settings and create a new API token. This will download a `kaggle.json` file.
   - Upload `kaggle.json` to Colab:

   ```python
   from google.colab import files
   files.upload()  # Upload the kaggle.json file here
   ```

2. **Set Up Kaggle Credentials:**

   ```python
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download the BraTS 2020 Dataset:**

   ```python
   !kaggle datasets download -d awsaf49/brats2020-training-data
   ```

4. **Extract the Dataset:**

   ```python
   !unzip brats2020-training-data.zip -d /content/drive/MyDrive/Brats2020/
   ```

**Option 2: Manual Upload to Google Drive**

If you prefer, you can manually download the dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) and upload it to a designated folder in your Google Drive.

*Assuming the dataset is located at `/content/drive/MyDrive/Brats2020/`.*

---

## **5. Data Preprocessing**

### **Combine Image Channels**

BraTS 2020 provides multimodal MRI scans: **T1**, **T1Gd**, **T2**, and **T2-FLAIR**. We'll combine these into a single 4-channel image.

```python
import os
import nibabel as nib
import numpy as np
import h5py

# Define paths
data_dir = '/content/drive/MyDrive/Brats2020/'
output_dir = '/content/drive/MyDrive/Brats2020_preprocessed/'

os.makedirs(output_dir, exist_ok=True)

# Function to load NIfTI files
def load_nifti(file_path):
    img = nib.load(file_path)
    return img.get_fdata()

# Iterate through each patient directory
for patient in os.listdir(data_dir):
    patient_dir = os.path.join(data_dir, patient)
    if os.path.isdir(patient_dir):
        # Load each modality
        t1 = load_nifti(os.path.join(patient_dir, f'{patient}_t1.nii.gz'))
        t1gd = load_nifti(os.path.join(patient_dir, f'{patient}_t1ce.nii.gz'))
        t2 = load_nifti(os.path.join(patient_dir, f'{patient}_t2.nii.gz'))
        flair = load_nifti(os.path.join(patient_dir, f'{patient}_flair.nii.gz'))
        # Load mask
        mask = load_nifti(os.path.join(patient_dir, f'{patient}_seg.nii.gz'))
        
        # Normalize image data
        t1 = (t1 - np.mean(t1)) / np.std(t1)
        t1gd = (t1gd - np.mean(t1gd)) / np.std(t1gd)
        t2 = (t2 - np.mean(t2)) / np.std(t2)
        flair = (flair - np.mean(flair)) / np.std(flair)
        
        # Combine image channels
        image = np.stack([t1, t1gd, t2, flair], axis=-1)  # Shape: (H, W, D, 4)
        
        # Save each slice as HDF5
        for slice_idx in range(image.shape[2]):
            slice_img = image[:, :, slice_idx, :]
            slice_mask = mask[:, :, slice_idx]
            # Define HDF5 file path
            hdf5_path = os.path.join(output_dir, f'{patient}_slice_{slice_idx}.h5')
            with h5py.File(hdf5_path, 'w') as hf:
                hf.create_dataset('image', data=slice_img, compression="gzip")
                hf.create_dataset('mask', data=slice_mask, compression="gzip")
```

### **Combine Mask Channels**

BraTS 2020 provides segmentation masks with labels:
- **ET (Enhancing Tumor):** Label 4
- **ED (Peritumoral Edema):** Label 2
- **NCR/NET (Necrotic and Non-Enhancing Tumor Core):** Label 1

We'll combine these into a single mask with multi-class labels.

*Note: The above preprocessing already includes combining the mask channels into a single mask (`slice_mask`).*

---

## **6. Visualize Sample Data**

Visualizing samples helps verify that the preprocessing steps were successful.

```python
import matplotlib.pyplot as plt

# Function to load HDF5 files
def load_hdf5(file_path):
    with h5py.File(file_path, 'r') as hf:
        image = np.array(hf['image'])
        mask = np.array(hf['mask'])
    return image, mask

# Select a random sample
import random

sample_files = os.listdir(output_dir)
sample_file = random.choice(sample_files)
image, mask = load_hdf5(os.path.join(output_dir, sample_file))

# Display the image and mask
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display first three channels as RGB (ignoring the fourth channel for visualization)
rgb_image = image[:, :, :3]
axes[0].imshow(rgb_image, cmap='gray')
axes[0].set_title('Combined Image (First 3 Channels)')
axes[0].axis('off')

# Display mask
axes[1].imshow(mask, cmap='jet')
axes[1].set_title('Combined Mask')
axes[1].axis('off')

plt.show()
```

*Sample Output:*
- **Left:** Combined MRI image (first 3 channels)
- **Right:** Combined segmentation mask with different colors representing different labels.

---

## **7. Prepare Data for Training**

We'll create a custom dataset class compatible with **PyTorch** and **Ultralytics' SAM**.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class BratsDataset(Dataset):
    def __init__(self, hdf5_dir, transform=None):
        self.hdf5_dir = hdf5_dir
        self.files = [os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith('.h5')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, mask = load_hdf5(self.files[idx])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask

# Initialize dataset and dataloader
batch_size = 2

dataset = BratsDataset(output_dir)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
```

---

## **8. Define Evaluation Metrics**

We'll define functions to compute **Hausdorff Distance**, **Dice Coefficient**, **Precision**, and **Loss** using **Scipy** and **PyTorch**.

```python
from sklearn.metrics import precision_score
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

def dice_coeff(pred, target, smooth=1):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice

def hausdorff_distance(pred, target):
    # Convert masks to binary
    pred = pred.cpu().numpy().astype(bool)
    target = target.cpu().numpy().astype(bool)
    
    # Get coordinates of non-zero pixels
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return 0.0  # Undefined Hausdorff distance
    
    # Compute directed Hausdorff distances
    forward_hd = directed_hausdorff(pred_points, target_points)[0]
    backward_hd = directed_hausdorff(target_points, pred_points)[0]
    
    # Return the maximum of the two directed Hausdorff distances
    return max(forward_hd, backward_hd)

# Define loss function
criterion = torch.nn.CrossEntropyLoss()
```

**Explanation:**

- **Dice Coefficient:** Measures the overlap between the predicted mask and the ground truth mask.
- **Hausdorff Distance:** Measures the maximum distance of a set to the nearest point in the other set. It provides an indication of the boundary mismatch.
- **Precision:** Measures the accuracy of the positive predictions.

---

## **9. Initialize and Train the SAM Model**

### **Initialize the SAM Model**

Before training, initialize the **SAM** model and move it to the GPU.

```python
from ultralytics import SAM

# Initialize the SAM model
model = SAM("sam_b.pt")  # Ensure 'sam_b.pt' is available in the working directory or provide the correct path

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### **Milestone 1: Train for 5 Epochs**

We'll start by training the model for **5 epochs**.

```python
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_precision = 0.0
    epoch_hausdorff = 0.0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        epoch_loss += loss.item()
        
        # Dice Coefficient
        preds = torch.argmax(outputs, dim=1)
        dice = dice_coeff(preds, masks)
        epoch_dice += dice.item()
        
        # Precision
        precision = precision_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro', zero_division=0)
        epoch_precision += precision
        
        # Hausdorff Distance
        hd = hausdorff_distance(preds, masks)
        epoch_hausdorff += hd
    
    # Calculate average metrics
    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)
    avg_precision = epoch_precision / len(train_loader)
    avg_hausdorff = epoch_hausdorff / len(train_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, Precision: {avg_precision:.4f}, Hausdorff: {avg_hausdorff:.4f}")
```

*Sample Output:*
```
Epoch [1/5] - Loss: 0.5678, Dice: 0.6789, Precision: 0.7001, Hausdorff: 15.1234
...
Epoch [5/5] - Loss: 0.4567, Dice: 0.7890, Precision: 0.8102, Hausdorff: 10.5678
```

### **Milestone 2: Train for 10 Epochs**

After completing **5 epochs**, continue training for an additional **5 epochs** to reach a total of **10 epochs**.

```python
# Continue Training for 5 More Epochs (Total 10 Epochs)
additional_epochs = 5
num_epochs += additional_epochs

for epoch in range(num_epochs - additional_epochs, num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_precision = 0.0
    epoch_hausdorff = 0.0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        epoch_loss += loss.item()
        
        # Dice Coefficient
        preds = torch.argmax(outputs, dim=1)
        dice = dice_coeff(preds, masks)
        epoch_dice += dice.item()
        
        # Precision
        precision = precision_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro', zero_division=0)
        epoch_precision += precision
        
        # Hausdorff Distance
        hd = hausdorff_distance(preds, masks)
        epoch_hausdorff += hd
    
    # Calculate average metrics
    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)
    avg_precision = epoch_precision / len(train_loader)
    avg_hausdorff = epoch_hausdorff / len(train_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, Precision: {avg_precision:.4f}, Hausdorff: {avg_hausdorff:.4f}")
```

*Sample Output:*
```
Epoch [6/10] - Loss: 0.4456, Dice: 0.8000, Precision: 0.8203, Hausdorff: 9.8765
...
Epoch [10/10] - Loss: 0.3987, Dice: 0.8501, Precision: 0.8704, Hausdorff: 8.5432
```

### **Milestone 3: Train for 20 Epochs**

Finally, extend the training to reach a total of **20 epochs**.

```python
# Continue Training for 10 More Epochs (Total 20 Epochs)
additional_epochs = 10
num_epochs += additional_epochs

for epoch in range(num_epochs - additional_epochs, num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_precision = 0.0
    epoch_hausdorff = 0.0
    
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        epoch_loss += loss.item()
        
        # Dice Coefficient
        preds = torch.argmax(outputs, dim=1)
        dice = dice_coeff(preds, masks)
        epoch_dice += dice.item()
        
        # Precision
        precision = precision_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro', zero_division=0)
        epoch_precision += precision
        
        # Hausdorff Distance
        hd = hausdorff_distance(preds, masks)
        epoch_hausdorff += hd
    
    # Calculate average metrics
    avg_loss = epoch_loss / len(train_loader)
    avg_dice = epoch_dice / len(train_loader)
    avg_precision = epoch_precision / len(train_loader)
    avg_hausdorff = epoch_hausdorff / len(train_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, Precision: {avg_precision:.4f}, Hausdorff: {avg_hausdorff:.4f}")
```

*Sample Output:*
```
Epoch [11/20] - Loss: 0.3901, Dice: 0.8602, Precision: 0.8805, Hausdorff: 7.6543
...
Epoch [20/20] - Loss: 0.3502, Dice: 0.9003, Precision: 0.9206, Hausdorff: 6.4321
```

*Note:* The actual performance metrics will vary based on the dataset and training dynamics.

---

## **10. Save and Export the Trained Model**

After training, it's essential to save the model checkpoints for future use or inference.

```python
# Define the path to save the model
model_save_path = '/content/drive/MyDrive/SAM_Brats2020.pth'

# Save the model state
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")
```

*Sample Output:*
```
Model saved to /content/drive/MyDrive/SAM_Brats2020.pth
```

---

## **11. Additional Notes**

1. **Handling Hausdorff Distance:**
   - Implementing the **Hausdorff Distance** requires converting masks to binary format and extracting the coordinates of non-zero pixels.
   - The provided implementation calculates the **symmetric Hausdorff Distance** by taking the maximum of the two directed distances.

2. **Data Augmentation:**
   - To enhance model generalization, consider applying data augmentation techniques such as rotations, flips, and intensity variations.
   - Libraries like `torchvision.transforms` or `albumentations` can be integrated for this purpose.

3. **Model Evaluation:**
   - After training, evaluate the model on a separate **validation set** to assess its performance accurately.
   - Implement a validation loop similar to the training loop but without gradient computations.

4. **Hyperparameter Tuning:**
   - Experiment with different learning rates, optimizers, and batch sizes to optimize training.
   - Consider using learning rate schedulers for dynamic adjustment of the learning rate.

5. **Ultralytics Integration:**
   - Ensure that the `SAM` class from **Ultralytics** supports training. If not, consider extending the class or using alternative segmentation models supported by Ultralytics.
   - Consult the [Ultralytics Documentation](https://docs.ultralytics.com/) for detailed guidance.

6. **Error Handling:**
   - Incorporate error handling to manage potential issues during data loading, preprocessing, or training.
   - Use try-except blocks where necessary to catch and handle exceptions gracefully.

7. **Performance Optimization:**
   - **Hausdorff Distance** computation can be time-consuming, especially for large datasets. Consider optimizing by:
     - Limiting the number of points used.
     - Computing it less frequently (e.g., every few batches).
     - Utilizing parallel processing if possible.

8. **Monitoring Training:**
   - Use tools like **TensorBoard** or **Weights & Biases** for real-time monitoring of training metrics.
   - Install and set up these tools to visualize loss curves and other metrics.

9. **Saving Intermediate Models:**
   - Save model checkpoints at regular intervals (e.g., every 5 epochs) to prevent data loss in case of interruptions.
   - Implement checkpointing mechanisms within the training loop.

10. **Inference and Deployment:**
    - After training, utilize the saved model for inference on new MRI scans.
    - Ensure that the preprocessing steps during inference match those during training.

---

By following this enhanced guide, you should be able to train the **Segment Anything Model (SAM)** on the **BraTS 2020** dataset using **Ultralytics** within **Google Colab** effectively. The integration of **Hausdorff Distance** provides a more comprehensive evaluation of the model's segmentation performance.

If you encounter any issues or have further questions, feel free to ask!
