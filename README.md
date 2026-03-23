# SelfCryoET

**J-Invariant Volume Shuffle for Self-Supervised Cryo-Electron Tomogram Denoising on Single Noisy Volume (WACV - Oral)**


## 🧪 Environment Setup

### 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the package in editable mode

```bash
pip install -e .
```

## 📦 Requirements

Recommended environment:

- Python 3.10 or 3.11
- PyTorch with CUDA support if training on GPU
- Linux or macOS

For exact GPU installation, install PyTorch from the official command shown on [pytorch.org](https://pytorch.org/get-started/locally/) if your CUDA version differs from the default wheel in `requirements.txt`.

## 🗂️ Data Directory Guideline

To keep training, preprocessing, inference, and evaluation scripts consistent, store your datasets with the following directory structure:

```text
data/
├── raw/
│   ├── shrec2020/
│   │   ├── sample_volume.npy
│   │   └── sample_gt.npy
│   ├── shrec2021/
│   │   ├── sample_volume.npy
│   │   └── sample_gt.npy
│   ├── polnet/
│   │   ├── sample_volume.npy
│   │   └── sample_gt.npy
│   └── real/
│       └── sample_volume.npy
├── processed/
│   ├── shrec2020/
│   ├── shrec2021/
│   ├── polnet/
│   └── real/
├── patches/
│   ├── shrec2020/
│   ├── shrec2021/
│   ├── polnet/
│   └── real/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Raw volume format

Current code supports:

- `.npy`
- `.pt`
- `.pth`

### Volume shape convention

Expected raw volume shape:

```text
(D, H, W)
```

The dataset loader will internally convert it to:

```text
(C, D, H, W)
```

with `C=1`.

### Notes on patch size

The current model uses `volume_unshuffle/shuffle` with factor `3`, so input patch sizes should be divisible by `3` at each downsampling stage. With the current 2-level U-shape implementation, it is safest to use dimensions divisible by:

```text
3 x 3 = 9
```

Examples:

- `108 x 108 x 108`
- `90 x 90 x 90`
- `72 x 72 x 72`


## 🚀 Reproduction Guideline

This section describes the recommended end-to-end process to reproduce the project workflow.

### Step 1. Prepare the dataset

Convert each tomographic volume into one of the supported formats and place it under `data/raw/...`.

For simulated datasets:

- noisy volume: `sample_volume.npy`
- clean reference: `sample_gt.npy`

For real datasets:

- only noisy volume is required

If your data is not already in `.npy`, convert it beforehand into a `float32` NumPy array with shape `(D, H, W)`.

### Step 2. Adjust config files

Update these files as needed:

- [configs/train.yaml]
- [configs/infer.yaml]
- [configs/dataset/shrec2020.yaml]
- [configs/dataset/shrec2021.yaml]
- [configs/dataset/polnet.yaml]
- [configs/dataset/real_data.yaml]

### Step 3. Preprocess a volume

Generate normalized, Gaussian-smoothed, bilateral-filtered, and edge-enhanced reference volumes:

```bash
python scripts/preprocess_volume.py \
  --input data/raw/shrec2020/sample_volume.npy \
  --output-dir data/processed/shrec2020 \
  --normalize \
  --gaussian-sigma 1.0 \
  --bilateral-kernel-size 5 \
  --bilateral-sigma-spatial 2.0 \
  --bilateral-sigma-intensity 0.1 \
  --prefix sample
```

### Step 4. Export patches

Patch export is optional, but useful for inspection and debugging:

```bash
python scripts/export_patches.py \
  --input data/raw/shrec2020/sample_volume.npy \
  --output-dir data/patches/shrec2020 \
  --patch-size 108 108 108 \
  --stride 54 54 54 \
  --normalize
```

### Step 5. Train the model

Train with the provided training config:

```bash
python scripts/train.py --config configs/train.yaml
```

You can also override values from the command line:

```bash
python scripts/train.py \
  --config configs/train.yaml \
  --override trainer.epochs=20 batch_size=1 trainer.checkpoint_dir=checkpoints/shrec2020
```

To save training history:

```bash
python scripts/train.py \
  --config configs/train.yaml \
  --save-history outputs/metrics/train_history.json
```

### Step 6. Run inference

```bash
python scripts/infer.py --config configs/infer.yaml
```

Example override:

```bash
python scripts/infer.py \
  --config configs/infer.yaml \
  --override input_path=data/raw/real/sample_volume.npy output_path=outputs/denoised/real_sample.npy
```

### Step 7. Evaluate the model

```bash
python scripts/evaluate.py --config configs/train.yaml
```

The current evaluation pipeline computes:

- validation loss
- PSNR
- SSIM

## ⚙️ Recommended Config Strategy

Use the config files like this:

- `configs/default.yaml`: common fallback settings
- `configs/train.yaml`: training-oriented settings
- `configs/infer.yaml`: inference-oriented settings
- `configs/dataset/*.yaml`: dataset-specific path and preprocessing templates

Recommended workflow:

1. copy a dataset config into your train or infer config
2. update the actual file paths
3. tune `patch_size`, `stride`, and `checkpoint_path`



## 📚 Citation

```bibtex
@inproceedings{liu2025mathcal,
  title={$$\backslash$mathcal $\{$J$\}$ $-Invariant Volume Shuffle for Self-Supervised Cryo-Electron Tomogram Denoising on Single Noisy Volume},
  author={Liu, Xiwei and Kassab, Mohamad and Xu, Min and Ho, Qirong},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={568--577},
  year={2025},
  organization={IEEE Computer Society}
}
```


