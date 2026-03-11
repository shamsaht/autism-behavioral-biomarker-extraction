# CCTV-Based Face & Emotion Analysis Pipeline for Autism Research

A multi-stage computer vision pipeline designed to analyse top-down CCTV footage of children in a clinical play-room environment. The system detects persons, extracts and aligns faces, classifies age demographics (child vs. adult), and predicts emotional states — all in support of autism spectrum disorder (ASD) behavioural research.



---

## Table of Contents

- [What the Project Does](#what-the-project-does)
- [Pipeline Architecture](#pipeline-architecture)
- [Requirements & Dependencies](#requirements--dependencies)
- [Installation](#installation)
- [How to Run the Code](#how-to-run-the-code)
- [File & Folder Descriptions](#file--folder-descriptions)
- [Examples of Usage](#examples-of-usage)
- [Known Limitations](#known-limitations)

---

## What the Project Does

This project implements an end-to-end pipeline that processes CCTV recordings from a clinical observation room. The pipeline performs the following stages:

1. **Frame Extraction** — Extracts frames from raw video files at a configurable interval.
2. **Person Detection** — Uses **YOLOv8** to detect persons in each frame.
3. **Face Detection & Alignment** — Uses **SCRFD** (via InsightFace) to locate facial landmarks and produce aligned face crops.
4. **Age Classification** — A fine-tuned **EfficientNet-B0** model classifies each face as `child` or `adult`.
5. **Emotion Classification** — **DeepFace** analyses child faces and maps emotions to a simplified 3-class system (`happy`, `sad`, `neutral`), with an extended 7-class variant also available.

The ultimate goal is to support autism researchers by quantifying emotional expression patterns of children during structured play sessions.

---

## Pipeline Architecture

```
Raw CCTV Videos
       │
       ▼
┌──────────────────────┐
│  Frame Extraction     │  (OpenCV, configurable FPS)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Person Detection     │  (YOLOv8-nano)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Face Detection &     │  (SCRFD / InsightFace)
│  Alignment            │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Age Classification   │  (EfficientNet-B0, fine-tuned)
│  child / adult        │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Emotion Classification│  (DeepFace → 3-class or 7-class)
│  happy / sad / neutral │
└──────────────────────┘
```

---

## Requirements & Dependencies

### Software

- **Python** 3.8+
- **Google Colab** (recommended for GPU execution) or a local machine with CUDA support

### Python Libraries

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Deep learning framework |
| `timm` | EfficientNet-B0 model loading |
| `ultralytics` | YOLOv8 person detection |
| `insightface`, `onnxruntime-gpu` | SCRFD face detection & alignment |
| `deepface` | Emotion analysis |
| `opencv-python` (`cv2`) | Image I/O and processing |
| `Pillow` | Image manipulation |
| `pandas` | Data handling and CSV I/O |
| `numpy` | Numerical operations |
| `matplotlib`, `seaborn` | Visualisations and figures |
| `scikit-learn` | Cross-validation utilities |
| `tqdm` | Progress bars |

### Hardware

- A **GPU** is strongly recommended (e.g., Colab T4 or better) for the YOLOv8, SCRFD, and EfficientNet inference steps.
- Approximately **10 GB** of Google Drive storage for video files, extracted frames, and face crops.

---

## Installation

### Option A — Google Colab (Recommended)

1. Upload the notebooks to your Google Drive.
2. Open any notebook in Google Colab.
3. Dependencies are installed automatically at the top of each notebook:

```python
!pip install ultralytics insightface onnxruntime-gpu deepface timm
```

4. Mount your Google Drive when prompted:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Option B — Local Setup

```bash
# Clone the repository
git clone https://github.com/shamsaht/autism-behavioral-biomarker-extraction.git
cd autism-behavioral-biomarker-extraction

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics insightface onnxruntime-gpu deepface timm
pip install opencv-python pillow pandas numpy matplotlib seaborn scikit-learn tqdm
```

---

## How to Run the Code

The pipeline is designed to be executed **sequentially** in Google Colab. Run the notebooks in the following order:

### Step 1 — Frame Extraction & Person Detection
```
Open: Frame_Extraction_Person_Detection.ipynb
```
- Extracts frames from raw videos at 1 FPS.
- Runs YOLOv8 to detect persons.
- Saves frames with detected persons and logs results to `person_detections.csv`.
- Supports **checkpointing** to resume after Colab timeouts.

### Step 2 — Manual Labelling
```
Open: Manual_Labeling_Tool.ipynb
```
- Interactive tool for manually labelling face crops as `child` or `adult`.
- Produces the labelled dataset needed to train/fine-tune the age classifier.

### Step 3 — Age Classification & Cross-Validation
```
Open: Age_Classification_Inference.ipynb        (inference on new data)
       Age_Classification_CrossValidation.ipynb  (5-fold cross-validation)
```
- `Age_Classification_Inference.ipynb` — Runs the fine-tuned EfficientNet-B0 model on all face crops and saves predictions to `age_classifications_v2.csv`.
- `Age_Classification_CrossValidation.ipynb` — Performs 5-fold cross-validation across videos with no overlap between folds.

### Step 4 — Emotion Classification
```
Open: Emotion_Classification.ipynb        (3-class: happy, sad, neutral)
       Emotion_Classification_7Class.ipynb (7-class: all DeepFace emotions)
```
- Runs DeepFace on faces classified as `child`.
- Maps emotions to the simplified 3-class system or the full 7-class system.
- Saves results to `emotion_classifications.csv`.

### Step 5 — Emotion Labelling (Optional)
```
Open: Emotion_Labeling_Tool.ipynb
```
- Interactive tool for manually verifying or correcting emotion predictions.

---

## File & Folder Descriptions

### Code Notebooks (Run in Order)

| File | Stage | Description |
|---|---|---|
| `Frame_Extraction_Person_Detection.ipynb` | 1 | Extracts frames from CCTV videos and detects persons using YOLOv8 |
| `Manual_Labeling_Tool.ipynb` | 2 | Interactive tool for labelling face crops as child/adult |
| `Age_Classification_Inference.ipynb` | 3 | Runs fine-tuned EfficientNet-B0 for age classification |
| `Age_Classification_CrossValidation.ipynb` | 3 | 5-fold cross-validation of the age classifier across videos |
| `Emotion_Classification.ipynb` | 4 | 3-class emotion classification (happy, sad, neutral) using DeepFace |
| `Emotion_Classification_7Class.ipynb` | 4 | 7-class emotion classification with detailed analysis |
| `Emotion_Labeling_Tool.ipynb` | 5 | Interactive tool for verifying/correcting emotion labels |



### Data Files

| File | Description |
|---|---|
| `person_detections.csv` | YOLOv8 person detection results per frame |
| `age_classifications_v2.csv` | Age classification predictions (child/adult) per face crop |
| `emotion_classifications.csv` | Emotion predictions per child face |
| `checkpoint.json` | Checkpointing state for resuming interrupted processing |


### Generated Outputs

| File | Description |
|---|---|
| `success_cases.png` | Composite figure showing pipeline success cases |
| `failure_cases.png` | Composite figure showing pipeline failure modes |

---

## Examples of Usage

### Running the Full Pipeline (Colab)

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Set your project path
BASE_PATH = "/content/drive/MyDrive/face_pipeline_project"

# 3. Run each notebook sequentially (see "How to Run" section above)
```



### Running Cross-Validation

```python
# In Age_Classification_CrossValidation.ipynb:
# The notebook automatically:
# 1. Scans all video folders in face_crops_aligned/
# 2. Splits videos into 5 non-overlapping folds
# 3. Runs inference on each fold
# 4. Saves results to cv_5fold_results.csv
```

---

## Known Limitations

| Limitation | Description |
|---|---|
| **Extreme top-down angles** | When subjects look directly at the floor, SCRFD cannot detect the face |
| **Severe occlusion** | Furniture or other people blocking the face causes misclassification |
| **Demographic edge cases** | Small adults may be misclassified as children at borderline confidence (~50%) |
| **Colab timeouts** | Long-running sessions may disconnect; checkpointing mitigates this |
| **Single camera angle** | Pipeline is optimised for a specific overhead CCTV setup |

![Pipeline Failure Cases](failure_cases.png)

---

## License

This project is part of an academic research internship focused on autism spectrum disorder behavioural analysis. Please contact the project maintainers for usage permissions.

---

## Acknowledgements

- **YOLOv8** by Ultralytics for person detection
- **InsightFace / SCRFD** for face detection and alignment
- **EfficientNet-B0** (via `timm`) for age classification
- **DeepFace** for emotion analysis
- Research conducted under clinical supervision for ASD behavioural study


