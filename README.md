# FG-MD-Remote-Sensing-Retrieval

FG-MD: Frequency-Guided Mutual Distillation Network
This is the official PyTorch implementation of the paper: "Learning Spatial-Semantic Consistency: A Frequency-Guided Mutual Distillation Framework for Remote Sensing Retrieval." 

Code Availability Note: The  pre-processing scripts for the VAIS dataset are currently available. The complete source code, including full training pipelines and pre-trained weights for all benchmarks, will be fully released upon the formal acceptance of the paper.

FG-MD is a lightweight and efficient framework for Cross-Modal Remote Sensing Image Retrieval (CMRSIR). By integrating frequency-domain perception and a symmetric mutual distillation strategy, it achieves state-of-the-art performance (0.8654 mAP on MRSSID) with only ~22M parameters, offering an optimal efficiency-accuracy trade-off for edge-device deployment.

Core Features
<img width="2133" height="1291" alt="image" src="https://github.com/user-attachments/assets/6e8c0268-cecb-4c7d-b370-26263d60e5b3" />


Frequency-Domain Full-Spectrum Perceiver (FDFSP): Utilizes Fast Fourier Transform (FFT) to recover high-frequency structural details (e.g., edges and textures) typically lost during downsampling. 

Adaptive Dynamic Filter (ADF): A content-adaptive semantic gate that strictly suppresses background clutter (e.g., sea waves) while preserving target-relevant features. 

Dual-Attention Mutual Distillation (DAMD): A symmetric collaborative learning strategy that ensures spatial-semantic consistency across modalities with zero inference overhead. 

Edge-Ready Performance: Based on a MobileNetV2 backbone, achieving real-time inference (48.6 FPS) on embedded platforms like NVIDIA Jetson NX.

Data Preparation

MRSSID: The primary benchmark for MS-PAN retrieval. 


VAIS: Visible-Infrared ship retrieval benchmark. Download: https://pan.baidu.com/s/15Rdv-_UJOGsAXeFb-8EgZw?pwd=q8yr (Password: q8yr)

To ensure a rigorous evaluation, the official training and testing split provided by the VAIS dataset is adopted, and identical hyperparameter configurations from the MRSSID experiments are maintained. This extension serves to verify the robustness of the model against substantial modality gaps between thermal and optical spectra under consistent experimental constraints.


IGS: Our self-constructed benchmark for indoor UAV reconnaissance.
