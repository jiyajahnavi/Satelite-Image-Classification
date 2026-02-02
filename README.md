
# Satellite Image Classification

A **machine learning system** that classifies satellite images into three categories and prepares them for transmission:

* **Horizon Detection**
* **Image Quality Evaluation**
* **Flare (sunburn) Detection**
* **Compression** 

The system provides **individual detectors** for each feature as well as a **unified pipeline** combining all three detectors.

---

## 📁 Project Structure

```
satellite_image_classification/
├── data/
│   ├── raw/                      # Original images (archival)
│   │   ├── earth/
│   │   ├── horizon/
│   │   ├── space/
│   │   └── sunburn/
│   │
│   └── classification_sets/
│       ├── horizon_detection/
│       │   ├── horizon/          # Label = 1 (horizon visible)
│       │   └── no_horizon/       # Label = 0 (no horizon)
│       ├── flare_detection/
│       │   ├── flare/            # Label = 1 (sun flare)
│       │   └── no_flare/         # Label = 0 (no flare)
│       └── quality_detection/
│           ├── good/             # Usable image
│           └── bad/              # Overexposed, blurred, etc.
├── models/                       # Saved model weights
├── results/                      # Visualization output directory
├── src/
│   ├── compression/              # Image compression module
│   ├── data/                     # Dataset and preprocessing scripts
│   ├── detection/                # Evaluation modules
│   ├── models/                   # Individual & unified classifiers
│   └── utils/                    # Shared utility functions
└── requirements.txt
```

---


## 🔄 Data Preprocessing

The preprocessing pipeline prepares the raw satellite images for training by:

* Resizing images to **224x224 or 256x256**
* Splitting into **train/validation/test sets**
* Applying **data augmentation** (rotation, flips, brightness/contrast)

**Command:**

```bash
python src/data/preprocess.py
```

---

## 🧠 Model Training

The system consists of **three binary classification models**:

1. **Horizon Detection Model** – Detects horizon in images
2. **Flare Detection Model** – Detects sun flares or glare
3. **Image Quality Detection Model** – Classifies images as good or bad

**Training Commands:**

```bash
python src/models/train_horizon_detector.py --batch_size 32 --img_size 224 --num_epochs 20 --learning_rate 0.001
python src/models/train_flare_detector.py --batch_size 32 --img_size 224 --num_epochs 20 --learning_rate 0.001
python src/models/train_quality_detector.py --batch_size 32 --img_size 224 --num_epochs 20 --learning_rate 0.001
```

---

## 📊 Evaluation and Visualization

**Individual Detector Evaluation:**

```bash
python -m src.detection.horizon_evaluation --image_path path/to/image.jpg --show
python -m src.detection.flare_evaluation --image_path path/to/image.jpg --show
python -m src.detection.quality_evaluation --image_path path/to/image.jpg --show
```

Each module:

* Loads the detector model
* Classifies input images
* Provides confidence scores
* Generates visualization (original + prediction) with color-coded indicators

---

## 🔄 Unified Pipeline

```bash
python -m src.models.unified_classifier --image_path path/to/image.jpg --visualize --save_viz results/output.jpg
```

* Processes input images through **all three detectors**
* Compresses **good-quality images** to ≤100KB
* Generates visualization and JSON output with confidence scores

---

## 📉 Image Compression

Images classified as **good quality** are compressed using a standalone module:

```bash
python -m src.compression.compress --input path/to/image.jpg --target_size 100
```

* Adaptive quality reduction to meet target size
* Falls back to resizing if needed
* Optimized JPEG compression

---

## 🔍 Example Output

```json
{
  "horizon": true,
  "horizon_confidence": 0.9568,
  "flare": false,
  "flare_confidence": 0.9999,
  "quality": "good",
  "quality_confidence": 0.9245,
  "compressed": {
    "path": "results/compressed_image.jpg",
    "compressed_size_kb": 83.45
  }
}
```

---

## 🧰 Tech Stack

* Python
* Scikit-learn
* OpenCV
* NumPy, Pandas
* Matplotlib

---

## 📄 License

This project is licensed under the **MIT License**. 
---
