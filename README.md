# NEXUS ISP v2
### Neural EXposure-aware Unified Smartphone Image Signal Processor

NEXUS ISP is a hybrid image enhancement system combining classical Image Signal Processing with deep learning-based image enhancement.

## Features
- Classical ISP pipeline
- Residual U-Net neural enhancement
- Adaptive fusion
- FP32 and FP16 inference
- Gradio web application

## Model Architecture
- Architecture: Residual U-Net
- Parameters: 726,643
- Training epochs: 20
- Best epoch: 18
- Training GPU: NVIDIA Tesla T4
- Inference checkpoint: 2.79 MB

## Experimental Results

| Method | PSNR (dB) | SSIM | MAE |
|---|---:|---:|---:|
| Degraded Input | 15.01 | 0.6514 | 0.15753 |
| Classical ISP | 13.07 | 0.6744 | 0.18469 |
| Adaptive Fusion | 18.18 | 0.8229 | 0.10460 |
| NEXUS ISP v2 | 26.35 | 0.9082 | 0.04093 |

These results are from the synthetic test dataset.

## Performance Benchmark
- Inference latency: 3.27 ms
- Throughput: 306.11 images/sec
- Peak GPU memory allocated: 33.81 MB
- Benchmark: NVIDIA Tesla T4, 128x128, batch size 1, FP16

## Dataset
- Total unique images: 788
- Training: 551
- Validation: 118
- Testing: 119

## Technology Stack
- Python
- PyTorch
- OpenCV
- NumPy
- Gradio
- CUDA
- Matplotlib

## Technical Limitations
- Neural enhancement operates in the RGB domain.
- RAW/DNG files are decoded to RGB before processing.
- Direct neural Bayer RAW-to-RGB reconstruction is not implemented.
- Adaptive fusion uses heuristic weighting.
- Real-world testing revealed colour consistency issues.

## Installation

```bash
git clone https://github.com/rachitha39/NEXUS-ISP-v2.git
cd NEXUS-ISP-v2
pip install torch torchvision opencv-python numpy gradio rawpy
python app.py
```

## Repository Structure

```text
NEXUS-ISP-v2/
├── app.py
├── .gitignore
└── deployment/
    ├── config.json
    ├── nexus_isp_v2_config.json
    └── nexus_isp_v2_inference.pth
```

## Future Improvements
- Improve colour consistency.
- Expand the training dataset.
- Improve adaptive fusion.
- Optimize inference for mobile and edge devices.
- Explore direct RAW-to-RGB reconstruction.

## GitHub Repository
https://github.com/rachitha39/NEXUS-ISP-v2
