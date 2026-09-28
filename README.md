# DTGS

## Qualitative Results

> **Note:** GIF files are large and may take a moment to load. Please be patient.

<p align="center"><b>demo1</b></p>

| GT                                 | Renders                                      |
| ---------------------------------- | -------------------------------------------- |
| ![demo1 GT](assets/demo1_gt.gif) | ![demo1 Renders](assets/demo1_renders.gif) |

<p align="center"><b>demo2</b></p>

| GT                               | Renders                                    |
| -------------------------------- | ------------------------------------------ |
| ![demo2 GT](assets/demo2_gt.gif) | ![demo2 Renders](assets/demo2_renders.gif) |

<p align="center"><b>demo3</b></p>

| GT                                 | Renders                                      |
| ---------------------------------- | -------------------------------------------- |
| ![demo3 GT](assets/demo3_gt.gif) | ![demo3 Renders](assets/demo3_renders.gif) |

### Download Full-Resolution Videos

For a clearer view of the results, full-resolution MP4 videos are available:

| Scene   | Ground Truth                            | Renders                                           |
| ------- | --------------------------------------- | ------------------------------------------------- |
| demo1   | [demo1_gt.mp4](assets/demo1_gt.mp4)     | [demo1_renders.mp4](assets/demo1_renders.mp4)     |
| demo2   | [demo2_gt.mp4](assets/demo2_gt.mp4)     | [demo2_renders.mp4](assets/demo2_renders.mp4)     |
| demo3   | [demo3_gt.mp4](assets/demo3_gt.mp4)     | [demo3_renders.mp4](assets/demo3_renders.mp4)     |


## Installation

### 1. Environment

- Python 3.8+
- CUDA 11.7+ (GPU required)
- PyTorch 1.13.1

### 2. Clone and install dependencies

```bash
git clone <repository-url>
cd DTGS

pip install -r requirements.txt
```

---

## Data Preparation

### Dynamic_LTR Dataset

Download the **Dynamic_LTR** dataset and place it under the `data/` directory.

> **Note:** The Dynamic_LTR dataset is currently being organized and will be released in mid-August.

```
DTGS/
├── data/
│   └── Dynamic_LTR/
│       ├── scene1/
│       │   ├── images/          # Thermal infrared frames
│       │   ├── depths/          # Depth maps (optional)
│       │   ├── masks/           # Binary masks (can be generated via make_mask.py)
│       │   ├── sparse/
│       │   │   └── 0/           # COLMAP sparse reconstruction
│       │   └── cameras.npz      # Camera parameters
│       ├── scene2/
│       └── ...
```

---

## Usage

### 1. Training

```bash
python train_thermal.py
```

By default, this trains on `Dynamic_LTR/scene1`. To switch scenes or datasets, edit the following lines in `train_thermal.py`:

```python
exp_name = "Dynamic_LTR/scene1"
source_path = "data/Dynamic_LTR/scene1"
```

Trained models and point clouds are saved to:

```
output/<exp_name>/
├── point_cloud/
│   └── iteration_<N>/
│       └── point_cloud.ply
├── input.ply
└── cfg_args
```

### 2. Rendering

Render images, depth maps, videos, and reconstructed point clouds from a trained model:

```bash
python render.py \
    --model_path output/Dynamic_LTR/scene1 \
    --skip_train \
    --skip_video \
    --reconstruct_test \
    --configs arguments/endonerf/default.py
```

**Rendering options:**

| Flag                  | Description                                  |
| --------------------- | -------------------------------------------- |
| `--model_path`        | Path to the trained model directory          |
| `--iteration`         | Checkpoint iteration to load (`-1` = latest) |
| `--skip_train`        | Skip rendering training views                |
| `--skip_test`         | Skip rendering test views                    |
| `--skip_video`        | Skip rendering video sequence                |
| `--reconstruct_train` | Reconstruct point clouds from training views |
| `--reconstruct_test`  | Reconstruct point clouds from test views     |
| `--reconstruct_video` | Reconstruct point clouds from video views    |
| `--configs`           | Hyperparameter config file                   |

Rendered outputs are saved to:

```
output/<exp_name>/
├── train/
│   └── ours_<iteration>/
│       ├── renders/      # Rendered images
│       ├── gt/           # Ground-truth images
│       ├── depth/        # Rendered depth maps
│       └── ...
├── test/
│   └── ours_<iteration>/
│       └── ...
├── video/
│   └── ours_<iteration>/
│       ├── ours_video.mp4
│       └── gt_video.mp4
└── reconstruct/          # Reconstructed point clouds
```

### 3. Evaluation

Compute PSNR, SSIM, LPIPS, and RMSE metrics on rendered results:

```bash
python metrics.py \
    --model_path output/Dynamic_LTR/scene1 \
    -p test
```

| Flag                  | Description                                   |
| --------------------- | --------------------------------------------- |
| `--model_path` / `-m` | Path(s) to the trained model directory        |
| `--phase` / `-p`      | Evaluation phase: `train`, `test`, or `video` |

Results are saved as JSON files under the model directory.

