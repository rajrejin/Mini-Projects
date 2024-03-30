import numpy as np
import cv2
from matplotlib import pyplot as plt
import albumentations as A
import os
import random

# Path to the dataset
dataset_path = "D:/FAU/4. WS 23/DSS/Exercises/My-Projects/Image Augmentation/Mini_BAGLS_dataset/"

# Get a list of all the image files in the dataset directory
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png') and not f.endswith('_seg.png')]

# Select a random image file
random_image_file = random.choice(image_files)

# Load the image and its corresponding mask
image = cv2.imread(os.path.join(dataset_path, random_image_file))
mask = cv2.imread(os.path.join(dataset_path, random_image_file.replace('.png', '_seg.png')), 0)

# Define an augmentation pipeline
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.GaussNoise(p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.PiecewiseAffine(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
], p=1, additional_targets={'image0': 'image'})

# Plot the original image and mask
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original image')
ax[1].imshow(mask, cmap='gray')
ax[1].set_title('Original mask')

# Apply transformations and plot
for i in range(4):
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']

    # Plot the transformed image
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(transformed_image, cmap='gray')
    ax[0].set_title('Transformed image {}'.format(i+1))
    ax[1].imshow(transformed_mask, cmap='gray')
    ax[1].set_title('Transformed mask {}'.format(i+1))
plt.show()