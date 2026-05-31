# 🗑️ PlasticSense — Smart Waste Sorting Bin

![Accuracy](https://img.shields.io/badge/Accuracy-97.7%25-brightgreen)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.977-brightgreen)
![Model](https://img.shields.io/badge/Model-MobileNetV3--Small-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red)

PlasticSense is a smart waste sorting prototype developed at **Sorbonne Université**. A camera analyzes the waste placed on a small circular tray, an AI model classifies it as **recyclable** or **non-recyclable**, and a servo motor automatically tilts the tray toward the right bin.

## Results

| Metric | Value |
|---|---|
| Best Val Accuracy | 98.4% |
| **Test Accuracy** | **97.7%** |
| F1 Score | 0.977 |
| Initial target | 90% |

### Progress across versions

| Version | Accuracy | What changed |
|---|---|---|
| V1 | 68.4% | Baseline model |
| V2 | 73.6% | First dataset cleanup |
| V3 | 84.6% | Added real iPhone photos |
| **V4** | **97.7%** | Full optimized pipeline |

## How it works

```
Waste placed on the tray
        ↓
    Camera captures image
        ↓
MobileNetV3-Small classifies
        ↓
Servo motor tilts the tray
        ↓
Recyclable ♻️  /  Non-recyclable 🚯
```

## Installation

```bash
git clone https://github.com/adnanegrb/PlasticSense.git
cd PlasticSense
pip install -r requirements.txt
```

## Usage

Train the model:
```bash
python train.py
```

Run inference on a single image:
```bash
python inference.py --image path/to/image.jpg
```

Output example:
```
♻️  RECYCLABLE
   Confidence : 98.3%
```

## Dataset

The dataset is not included in this repo. See `dataset/README.md` for download instructions.

8,832 images split into two classes: recyclable (plastic, paper, cardboard, metal) and non-recyclable (biological waste, mixed trash). Split: 80% train / 10% val / 10% test.

## Model

MobileNetV3-Small pretrained on ImageNet, with the last 3 blocks unfrozen for fine-tuning. Trained with AdamW optimizer, CosineAnnealingLR scheduler, WeightedRandomSampler, and Early Stopping (patience = 3).

## Hardware

Raspberry Pi 4, CSI camera module, servo motor, 5V power supply.

## Author

**Mohammed Adnane Garab** — Sorbonne Université
[GitHub](https://github.com/adnanegrb)

## License

MIT License
