# FG-MD-Remote-Sensing-Retrieval

FG-MD: Frequency-Guided Mutual Distillation Network
This is the official PyTorch implementation of the paper: "Learning Spatial-Semantic Consistency: A Frequency-Guided Mutual Distillation Framework for Remote Sensing Retrieval." 

Code Availability Note: The  pre-processing scripts for the VAIS dataset are currently available. The complete source code, including full training pipelines and pre-trained weights for all benchmarks, will be fully released upon the formal acceptance of the paper.

FG-MD is a lightweight and efficient framework for Cross-Modal Remote Sensing Image Retrieval (CMRSIR). By integrating frequency-domain perception and a symmetric mutual distillation strategy, it achieves state-of-the-art performance (0.8654 mAP on MRSSID) with only ~22M parameters, offering an optimal efficiency-accuracy trade-off for edge-device deployment.

Core Features

Frequency-Domain Full-Spectrum Perceiver (FDFSP): Utilizes Fast Fourier Transform (FFT) to recover high-frequency structural details (e.g., edges and textures) typically lost during downsampling. 

Adaptive Dynamic Filter (ADF): A content-adaptive semantic gate that strictly suppresses background clutter (e.g., sea waves) while preserving target-relevant features. 

Dual-Attention Mutual Distillation (DAMD): A symmetric collaborative learning strategy that ensures spatial-semantic consistency across modalities with zero inference overhead. 

Edge-Ready Performance: Based on a MobileNetV2 backbone, achieving real-time inference (48.6 FPS) on embedded platforms like NVIDIA Jetson NX.

Data Preparation

MRSSID: The primary benchmark for MS-PAN retrieval. 


VAIS: Visible-Infrared ship retrieval benchmark. Download: https://pan.baidu.com/s/15Rdv-_UJOGsAXeFb-8EgZw?pwd=q8yr (Password: q8yr)

Note on Reproducibility: While some works claim ~99% mAP on VAIS, those results are often non-open-source. We provide a verifiable baseline with an mAP of ~0.78 (at 24 bits) to ensure academic transparency.


IGS: Our self-constructed benchmark for indoor UAV reconnaissance.
