
import os
import json
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import rawpy

from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = BASE_DIR / "deployment"

# Preserve v1 and use v2 as the default model
V1_MODEL_PATH = DEPLOY_DIR / "nexus_isp_inference.pth"
V2_MODEL_PATH = DEPLOY_DIR / "nexus_isp_v2_inference.pth"

MODEL_PATH = V2_MODEL_PATH
CONFIG_PATH = DEPLOY_DIR / "nexus_isp_v2_config.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RAW_EXTENSIONS = {
    ".dng", ".cr2", ".cr3", ".nef",
    ".arw", ".rw2", ".orf", ".raf"
}

OUTPUT_DIR = BASE_DIR / "outputs" / "application"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. RESIDUAL U-NET ARCHITECTURE
# Must match the architecture used during Phase 4 training.
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class NexusResidualUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()

        self.enc1 = DoubleConv(3, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(
            base * 8, base * 4, 2, stride=2
        )
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2
        )
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2
        )
        self.dec1 = DoubleConv(base * 2, base)

        self.output = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        original = x

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        correction = 0.5 * torch.tanh(self.output(d1))

        return torch.clamp(original + correction, 0, 1)


# ============================================================
# 3. LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = NexusResidualUNet(base=16).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

# Support both full training checkpoints and weights-only checkpoints
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print("NEXUS ISP v2 loaded successfully")
print("Model:", MODEL_PATH.name)
print("Device:", DEVICE)
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# 4. IMAGE LOADING
# ============================================================

def load_image(path):
    path = Path(path)

    if path.suffix.lower() in RAW_EXTENSIONS:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                gamma=(2.222, 4.5),
                output_color=rawpy.ColorSpace.sRGB
            )

        image = rgb.astype(np.float32) / 255.0
        input_type = "Camera RAW"

    else:
        image = Image.open(path).convert("RGB")
        image = np.asarray(image).astype(np.float32) / 255.0
        input_type = "RGB image"

    return np.clip(image, 0, 1), input_type


# ============================================================
# 5. NEURAL ISP
# ============================================================

@torch.inference_mode()
def neural_isp(rgb, precision="FP32"):
    image = np.clip(
        np.asarray(rgb, dtype=np.float32), 0, 1
    )

    tensor = torch.from_numpy(
        image
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    _, _, h, w = tensor.shape

    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    tensor = F.pad(
        tensor,
        (0, pad_w, 0, pad_h),
        mode="reflect"
    )

    if precision == "FP16" and DEVICE.type == "cuda":
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            output = model(tensor)
    else:
        output = model(tensor)

    output = output[:, :, :h, :w]

    return (
        output.squeeze(0)
        .float()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .clip(0, 1)
    )


# ============================================================
# 6. CLASSICAL ISP
# ============================================================

def classical_isp(rgb):
    img = np.clip(
        np.asarray(rgb, dtype=np.float32), 0, 1
    )

    channel_means = np.mean(img, axis=(0, 1))
    gray = np.mean(channel_means)

    gains = gray / (channel_means + 1e-6)
    gains = np.clip(gains, 0.80, 1.25)

    img = np.clip(
        img * gains.reshape(1, 1, 3), 0, 1
    )

    ccm = np.array([
        [1.08, -0.04, -0.04],
        [-0.03, 1.06, -0.03],
        [-0.02, -0.05, 1.07]
    ], dtype=np.float32)

    img = np.clip(img @ ccm.T, 0, 1)

    img8 = (img * 255).astype(np.uint8)

    img8 = cv2.bilateralFilter(
        img8, d=5, sigmaColor=35, sigmaSpace=35
    )

    gray_image = cv2.cvtColor(
        img8, cv2.COLOR_RGB2GRAY
    )

    brightness = float(np.mean(gray_image) / 255.0)
    gamma = 0.95 if brightness > 0.45 else 0.85

    img = (img8.astype(np.float32) / 255.0) ** gamma

    lab = cv2.cvtColor(
        (img * 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    result = cv2.cvtColor(
        lab, cv2.COLOR_LAB2RGB
    ).astype(np.float32) / 255.0

    blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
    result = cv2.addWeighted(
        result, 1.15, blurred, -0.15, 0
    )

    return np.clip(result, 0, 1)


# ============================================================
# 7. SCENE ANALYSIS AND ADAPTIVE FUSION
# ============================================================

def analyze_scene(rgb):
    img8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    gray = cv2.cvtColor(
        img8, cv2.COLOR_RGB2GRAY
    ).astype(np.float32) / 255.0

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    median = cv2.medianBlur(
        (gray * 255).astype(np.uint8), 3
    ).astype(np.float32) / 255.0

    residual = gray - median

    noise = float(
        np.median(np.abs(residual)) / 0.6745
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "noise_estimate": noise
    }


def fusion_weight(scene):
    alpha = 0.45

    noise_factor = np.clip(
        scene["noise_estimate"] / 0.08, 0, 1
    )

    darkness_factor = np.clip(
        (0.40 - scene["brightness"]) / 0.40,
        0, 1
    )

    alpha += 0.20 * noise_factor
    alpha += 0.15 * darkness_factor

    return float(np.clip(alpha, 0.25, 0.75))


# ============================================================
# 8. MAIN PROCESSING FUNCTION
# ============================================================

def process_image(file_path, mode, precision):
    if not file_path:
        raise gr.Error("Please upload an image first.")

    start_time = time.perf_counter()

    try:
        rgb, input_type = load_image(file_path)

        if mode == "Classical ISP":
            result = classical_isp(rgb)
            alpha = None

        elif mode == "Neural ISP":
            result = neural_isp(rgb, precision)
            alpha = None

        else:
            classical = classical_isp(rgb)
            neural = neural_isp(rgb, precision)

            scene = analyze_scene(rgb)
            alpha = fusion_weight(scene)

            result = (
                alpha * neural +
                (1 - alpha) * classical
            )

            result = np.clip(result, 0, 1)

        scene = analyze_scene(rgb)
        elapsed = time.perf_counter() - start_time

        # Save enhanced result.
        filename = (
            f"nexus_enhanced_{int(time.time() * 1000)}.png"
        )

        output_path = OUTPUT_DIR / filename

        Image.fromarray(
            (result * 255).astype(np.uint8)
        ).save(output_path)

        stats = [
            ["Input type", input_type],
            ["Image dimensions", f"{rgb.shape[1]} × {rgb.shape[0]}"],
            ["Processing mode", mode],
            ["Precision", precision],
            ["Brightness", f"{scene['brightness']:.4f}"],
            ["Contrast", f"{scene['contrast']:.4f}"],
            ["Estimated noise", f"{scene['noise_estimate']:.4f}"],
            ["Processing time", f"{elapsed:.3f} seconds"],
            ["Device", str(DEVICE)]
        ]

        if alpha is not None:
            stats.extend([
                ["Neural contribution", f"{alpha:.2f}"],
                ["Classical contribution", f"{1-alpha:.2f}"]
            ])

        return (
            Image.fromarray((rgb * 255).astype(np.uint8)),
            Image.fromarray((result * 255).astype(np.uint8)),
            stats,
            str(output_path)
        )

    except gr.Error:
        raise

    except Exception as error:
        raise gr.Error(
            f"Image processing failed: {str(error)}"
        )


# ============================================================
# 9. GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title="NEXUS ISP — Intelligent Smartphone ISP",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # NEXUS ISP
        ### Neural EXposure-aware Unified Smartphone Image Signal Processor

        Enhance photographs using classical image processing,
        a trained Residual U-Net, or scene-aware adaptive fusion.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):

            input_file = gr.File(
                label="Upload photograph",
                file_types=[
                    ".jpg", ".jpeg", ".png",
                    ".dng", ".cr2", ".cr3", ".nef",
                    ".arw", ".rw2", ".orf", ".raf"
                ],
                type="filepath"
            )

            mode = gr.Radio(
                choices=[
                    "Classical ISP",
                    "Neural ISP",
                    "Adaptive Fusion"
                ],
                value="Adaptive Fusion",
                label="ISP Processing Mode"
            )

            precision = gr.Radio(
                choices=["FP32", "FP16"],
                value="FP32",
                label="Inference Precision",
                info="FP16 requires a compatible CUDA GPU."
            )

            process_button = gr.Button(
                "Enhance Image",
                variant="primary"
            )

            gr.Markdown(
                """
                **Supported inputs:** JPEG, PNG and supported camera RAW files.

                RAW files are decoded to RGB before enhancement.
                """
            )

        with gr.Column(scale=2):
            with gr.Row():
                original_output = gr.Image(
                    label="Original Image",
                    type="pil"
                )

                enhanced_output = gr.Image(
                    label="Enhanced Image",
                    type="pil"
                )

            stats_output = gr.Dataframe(
                headers=["Property", "Value"],
                label="Processing Information",
                interactive=False
            )

            download_output = gr.File(
                label="Download Enhanced Image"
            )

    process_button.click(
        fn=process_image,
        inputs=[input_file, mode, precision],
        outputs=[
            original_output,
            enhanced_output,
            stats_output,
            download_output
        ]
    )

    gr.Markdown(
        """
        ---
        **NEXUS ISP | AI-powered computational photography prototype**

        Developed using Python, OpenCV, PyTorch, LibRaw and Gradio.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
