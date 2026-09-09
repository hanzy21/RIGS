# RIGS

Official implementation of the ECCV 2026 paper **RIGS: Radar-Informed Gaussian
Splatting for Uncertainty-Aware 3D Occupancy and Motion Prediction**.

## Data

RIGS uses the K-Radar dataset. Training requires synchronized front-camera
images, calibration and ego-pose data, semantic occupancy annotations, and the
preprocessed K-Radar 4D radar tensor data. 

## Environment

Use Python 3.10, PyTorch 2.1.2 with CUDA 11.8, and an NVIDIA CUDA GPU. Install
the matching PyTorch build first, then install the remaining dependencies and
compile the CUDA extensions:

```bash
pip install -r requirements.txt
bash scripts/build_extensions.sh
```

The Python dependencies include MMCV 2.1.0, MMEngine 0.10.7, MMSegmentation
1.0.0, MMDetection 3.0.0, MMDetection3D 1.1.1, and spconv 2.3.8.

## Usage

Build a manifest and K-Radar runtime index:

```bash
python tools/build_manifest.py \
  --data-root /path/to/kradar \
  --resources-root /path/to/K-Radar/resources \
  --output-dir data/manifests

python tools/compile_index.py \
  --manifest data/manifests/train.jsonl \
  --data-root /path/to/kradar \
  --resources-root /path/to/K-Radar/resources \
  --output data/index/kradar_infos_train.pkl
```

Preprocess the radar tensors, then set the data and index locations before
training:

```bash
python tools/preprocess_radar.py --help

export KRADAR_DATA_ROOT=/path/to/kradar
export RIGS_INDEX_ROOT=/path/to/index
export RIGS_BACKBONE_CHECKPOINT=/path/to/backbone_initialization.pth

python train.py --config config/rigs_kradar.py --work-dir work_dirs/rigs
```

For sequential inference, create an inference manifest without occupancy paths,
compile its index, and run:

```bash
python tools/make_inference_manifest.py \
  --manifest data/manifests/test.jsonl \
  --output data/manifests/test_nogt.jsonl

python tools/compile_index.py \
  --manifest data/manifests/test_nogt.jsonl \
  --data-root /path/to/kradar \
  --resources-root /path/to/K-Radar/resources \
  --output data/index/kradar_infos_test_nogt.pkl

python tools/infer.py \
  --config config/rigs_kradar.py \
  --checkpoint /path/to/model.pth \
  --manifest data/manifests/test_nogt.jsonl \
  --index data/index/kradar_infos_test_nogt.pkl \
  --data-root /path/to/kradar \
  --output-dir outputs/inference
```

Use `tools/visualize.py` to render saved inference outputs.
