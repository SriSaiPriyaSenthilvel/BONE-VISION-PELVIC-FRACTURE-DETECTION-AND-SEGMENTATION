# BONE-VISION: Pelvic Fracture Detection & Segmentation Using Deep Learning


## Abstract

Pelvic fractures are among the most complex and high-risk injuries in orthopedic trauma care. They involve overlapping bone structures, low-contrast radiographs, and high anatomical variability, making manual interpretation difficult even for experienced radiologists. Early detection is essential to prevent internal bleeding, organ damage, and long-term mobility issues.
BONE-VISION is an AI-powered diagnostic system that integrates:

### CNN-based fracture classification

### YOLOv8 object detection for precise localization

### U-Net segmentation for pixel-level fracture masking

### Advanced preprocessing (CLAHE, Gaussian smoothing, noise reduction)

The pipeline processes X-rays in milliseconds and provides an end-to-end automated workflow for fracture identification, localization, and segmentation. This system significantly reduces radiologist workload and enables fast, objective, and highly scalable medical imaging support suitable for hospitals, trauma centers, and telemedicine platforms.

## Key Features

✔ Automated pelvic fracture detection
✔ Real-time bounding box localization using YOLOv8
✔ Pixel-level segmentation with U-Net
✔ Robust preprocessing for low-contrast X-rays
✔ Streamlit-powered intuitive UI
✔ Lightweight, fast inference backend
✔ Deployable on HuggingFace Spaces

## Tech Stack

Languages: Python

Deep Learning: TensorFlow, Keras, PyTorch

Models: U-Net, YOLOv8, CNN

Frontend: Streamlit

Deployment: HuggingFace Spaces / Local

Tools: OpenCV, NumPy, Matplotlib

📂 Project Structure
 BONE-VISION
 ┣ 📂 src
 │  ┣ streamlit_app.py
 │  ┣ model_weights.h5
 │  ┣ unet.py
 │  ┣ preprocessing.py
 │  ┗ utils.py
 ┣ 📂 models
 │  ┣ model.h5
 ┣ requirements.txt
 ┣ README.md

## Preprocessing Techniques

Noise Filtering

Bone Edge Enhancement

Adaptive Thresholding

Normalization (0–1)

## Model Architectures

Input: 128×128 grayscale image

Convolution → ReLU → Pooling → Dense

Output: Fracture / No Fracture

Trained on pelvic fracture bounding boxes

Outputs: Class + Confidence + Bounding Box

U-Net – Fracture Region Segmentation

Encoder–Decoder architecture

Skip connections for high-resolution feature recovery

Output: 128×128 segmentation mask

## Usage Instructions

Upload a pelvic X-ray image

The image is preprocessed

CNN predicts fracture probability

U-Net generates a segmentation mask

Results displayed instantly via Streamlit

## Output:
Detection Output

![WhatsApp Image 2025-12-05 at 18 34 20_25b31a28](https://github.com/user-attachments/assets/7aa550ce-f6e5-4958-ade0-7ebdb1d85a2c)


![WhatsApp Image 2025-12-05 at 18 34 20_258b19fb](https://github.com/user-attachments/assets/ea434d5a-90e2-4fb8-a2de-8aac31f2d6a9)
