# Real-Time-GPU-Accelerated-Image-Processing-Using-CUDA-CuPy-and-OpenCV

## Description
This project demonstrates GPU acceleration using CUDA through CuPy and OpenCV in Google Colab. The project compares CPU and GPU execution times for image processing operations including grayscale conversion, Gaussian blur, and edge detection.

## Features
```
CUDA GPU acceleration
Image filtering
CPU vs GPU comparison
Performance visualization
```

## Technologies
```
Python
CUDA
CuPy
OpenCV
Google Colab
```

## How to Run
```
Open Google Colab
Enable GPU runtime
Run all cells
Upload image when prompted
```

## Program
```
# =====================================================
# GPU Accelerated Image Processing using CUDA
# =====================================================

# Install libraries
!pip install cupy-cuda12x opencv-python matplotlib -q

# Imports
import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
from google.colab import files

# =====================================================
# Upload Image
# =====================================================

uploaded = files.upload()

# Get uploaded file name
image_path = list(uploaded.keys())[0]

# Load image
image = cv2.imread(image_path)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image Loaded Successfully")

# =====================================================
# CPU Processing
# =====================================================

start_cpu = time.time()

# Grayscale
gray_cpu = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Gaussian Blur
blur_cpu = cv2.GaussianBlur(gray_cpu, (15,15), 0)

# Edge Detection
edges_cpu = cv2.Canny(blur_cpu, 100, 200)

end_cpu = time.time()

cpu_time = end_cpu - start_cpu

# =====================================================
# GPU Processing using CuPy
# =====================================================

start_gpu = time.time()

# Move image to GPU
gpu_image = cp.asarray(image_rgb)

# Convert to grayscale manually
gray_gpu = (
    0.2989 * gpu_image[:,:,0] +
    0.5870 * gpu_image[:,:,1] +
    0.1140 * gpu_image[:,:,2]
)

gray_gpu = gray_gpu.astype(cp.uint8)

# Move back to CPU for OpenCV operations
gray_gpu_cpu = cp.asnumpy(gray_gpu)

# Gaussian Blur
blur_gpu = cv2.GaussianBlur(gray_gpu_cpu, (15,15), 0)

# Edge Detection
edges_gpu = cv2.Canny(blur_gpu, 100, 200)

end_gpu = time.time()

gpu_time = end_gpu - start_gpu

# =====================================================
# Results
# =====================================================

print("\n==============================")
print("CPU Processing Time :", cpu_time, "seconds")
print("GPU Processing Time :", gpu_time, "seconds")
print("==============================")

# =====================================================
# Visualization
# =====================================================

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(edges_cpu, cmap='gray')
plt.title("CPU Edge Detection")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(edges_gpu, cmap='gray')
plt.title("GPU Edge Detection")
plt.axis("off")

plt.show()

# =====================================================
# Performance Comparison Graph
# =====================================================

methods = ['CPU', 'GPU']
times = [cpu_time, gpu_time]

plt.figure(figsize=(6,5))
plt.bar(methods, times)

plt.ylabel("Execution Time (seconds)")
plt.title("CPU vs GPU Performance")

plt.show()

# =====================================================
# Save Outputs
# =====================================================

cv2.imwrite("cpu_output.png", edges_cpu)
cv2.imwrite("gpu_output.png", edges_gpu)

print("\nOutput images saved successfully.")

# =====================================================
# GPU Information
# =====================================================

print("\nGPU Device Info:")
print(cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
```

## Output
<img width="1004" height="627" alt="image" src="https://github.com/user-attachments/assets/de3041fa-c1f9-491f-85d0-1107ab4c9175" />

## Result
GPU processing showed improved execution speed compared to CPU processing.
