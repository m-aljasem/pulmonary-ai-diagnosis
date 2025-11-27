# 🫁 NIH Chest X-ray Multi-Classification

Deep learning system for **multi-label detection** of 14 thoracic conditions from chest X‑ray images using DenseNet121.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 👤 Author

- **Name**: Mohamad AlJasem, MD MPH MSc  
- **Email**: [mohamad@aljasem.eu.org](mailto:mohamad@aljasem.eu.org)  
- **GitHub**: [github.com/m-aljasem](https://github.com/m-aljasem)  
- **Website**: [aljasem.eu.org](https://aljasem.eu.org)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Conditions](#-conditions)
- [Exported Weights](#-exported-weights)
- [License](#-license)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

This project provides a **multi-label classifier** for chest X‑rays based on the **NIH ChestX-ray14** dataset.  
Given a single X‑ray, the model outputs probabilities for up to **14 different pathologies**.

The project includes:

- A training pipeline around a DenseNet121 backbone
- A Streamlit app for interactive classification
- Exported weights for portable deployment

---

## ✨ Features

- **DenseNet121 transfer learning** on chest X‑ray images
- **Multi-label** output (sigmoid activation per class)
- Threshold-based condition display in the UI
- Pathology-wise performance reporting (once training is enabled)

---

## 🛠 Tech Stack

- Python 3.8+
- TensorFlow / Keras
- DenseNet121
- Streamlit

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Dev tools:

```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Quick Start

### 1️⃣ Train the Model

```bash
cd pulmonary-ai-diagnosis
python src/train.py
```

When the data pipeline is wired, this will save best weights to:

```text
models/chest_xray_model.h5
```

### 2️⃣ Run the Streamlit App

```bash
cd pulmonary-ai-diagnosis
streamlit run app.py
```

Upload a chest X‑ray and see which conditions are predicted above the chosen threshold.

---

## 🧑‍💻 Usage

### 🌐 Web App

```bash
streamlit run app.py
```

The app:
- Builds the DenseNet121-based model
- Loads weights from `models/chest_xray_model.h5` if present
- Displays conditions with probability > 0.5 (or threshold you can adapt)

### 🧬 Programmatic Usage

```python
from src.model import build_chest_xray_model
import numpy as np

CLASSES = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
           'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
           'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

model = build_chest_xray_model()
model.load_weights("models/chest_xray_model.h5")  # after training

# img_preprocessed: (1, 224, 224, 3)
pred = model.predict(img_preprocessed, verbose=0)[0]
detected = [(cls, p) for cls, p in zip(CLASSES, pred) if p > 0.5]
```

---

## 🗂 Project Structure

```text
pulmonary-ai-diagnosis/
├── app.py                    # Streamlit app
├── config/
├── data/                     # NIH ChestX-ray14 CSV + images
├── docs/
├── experiments/
├── models/                   # Saved weights (chest_xray_model.h5)
├── notebooks/
├── scripts/
├── src/
│   ├── __init__.py
│   └── model.py              # build_chest_xray_model()
└── tests/
```

---

## 🩺 Conditions

The model predicts probabilities for the following 14 labels:

- Atelectasis  
- Consolidation  
- Infiltration  
- Pneumothorax  
- Edema  
- Emphysema  
- Fibrosis  
- Effusion  
- Pneumonia  
- Pleural Thickening  
- Cardiomegaly  
- Nodule  
- Mass  
- Hernia  

---

## 📦 Exported Weights

- Training script saves to:

```text
../models/chest_xray_model.h5
```

- Streamlit app loads from:

```text
models/chest_xray_model.h5
```

Copy `models/chest_xray_model.h5` along with the project to deploy elsewhere.

---

## 📄 License

Licensed under the **MIT License**.  
See `LICENSE` for details.

---

## 🏥 Disclaimer

> This project is for **research and educational purposes only**.  
> It must **not** be used for clinical diagnosis, triage, or treatment decisions.

