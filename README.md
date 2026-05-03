# 🌿 AgriScan AI — Plant Disease Detection with CNN + Groq LLM

> **AI-powered plant disease detection** combining a PyTorch CNN with Groq LLM (LLaMA 3.3-70B) to deliver instant disease diagnosis, treatment plans, and prevention strategies — aligned with **UN SDG 2: Zero Hunger**.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8-red?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-1.1.2-green?logo=flask)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3--70B-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![SDG 2](https://img.shields.io/badge/UN_SDG-2_Zero_Hunger-yellow)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [SDG 2 — Zero Hunger Alignment](#-sdg-2--zero-hunger-alignment)
- [Architecture](#-architecture)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Run Locally](#-run-locally)
- [Run with Docker](#-run-with-docker)
- [API Usage](#-api-usage)
- [DockerHub](#-dockerhub)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌱 Project Overview

**AgriScan AI** is a production-ready plant disease detection system that:

1. **CNN Layer** — Uses a custom PyTorch Convolutional Neural Network trained on the PlantVillage dataset to classify leaf images into **39 disease categories** across 14 crop species.

2. **LLM Layer (Agentic AI)** — After CNN classification, a **Groq-powered LLaMA 3.3-70B** model acts as an agricultural advisor, generating:
   - Human-readable disease diagnosis
   - Step-by-step treatment plans
   - Preventive measures
   - SDG 2 impact analysis

3. **Inference Serving** — Flask REST API serves predictions via `/predict` endpoint.

4. **Containerized** — Fully Dockerized for reproducible deployment.

---

## 🎯 SDG 2 — Zero Hunger Alignment

This project directly supports the **United Nations Sustainable Development Goal 2: Zero Hunger** by:

| Aspect | How AgriScan AI Contributes |
|---|---|
| **Early Detection** | Identifies plant diseases before they spread, reducing crop loss by up to 30-50%. |
| **Accessible Technology** | Provides AI-powered diagnostics via a simple web upload — no agronomy degree required. |
| **Actionable Advice** | LLM generates treatment steps that smallholder farmers can follow immediately. |
| **Preventive Agriculture** | Prevention tips help farmers adopt sustainable practices, reducing future disease incidence. |
| **Food Security** | By protecting crop yields, the system contributes to stable food supply chains. |
| **Sustainable Agriculture** | Promotes targeted treatment over blanket pesticide use, supporting environmental sustainability. |

---

## 🏗 Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  User/Client│───▶│  Flask REST API   │───▶│  PyTorch CNN Model  │
│  (curl/web) │    │  (app.py:5000)    │    │  (39-class classifier│
└─────────────┘    └────────┬─────────┘    │   PlantVillage)     │
                            │              └──────────┬──────────┘
                            │                         │
                            │              ┌──────────▼──────────┐
                            │              │  Groq LLM Agent     │
                            │              │  (LLaMA 3.3-70B)    │
                            │              │  groq_agent.py      │
                            │              └──────────┬──────────┘
                            │                         │
                            ▼                         ▼
                   ┌─────────────────────────────────────┐
                   │  Combined JSON Response              │
                   │  CNN prediction + LLM diagnosis      │
                   │  + treatment + prevention + SDG2     │
                   └─────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────┐
│               Docker Container                   │
│  ┌───────────────────────────────────────────┐   │
│  │  python:3.10-slim                         │   │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  │   │
│  │  │ Flask   │  │ PyTorch  │  │ Groq    │  │   │
│  │  │ Server  │──│ CNN      │──│ Agent   │  │   │
│  │  │ :5000   │  │ Model    │  │ (LLM)   │  │   │
│  │  └─────────┘  └──────────┘  └─────────┘  │   │
│  └───────────────────────────────────────────┘   │
│                  Port 5000                        │
└─────────────────────────────────────────────────┘
```

---

## ✨ Features

- 🔬 **39-class plant disease classification** across 14 crop species
- 🤖 **Agentic AI** — Groq LLM provides expert-level diagnosis
- 🌐 **REST API** — JSON responses for easy integration
- 🖥️ **Web UI** — Upload leaf images and get results visually
- 🐳 **Docker** — One-command deployment
- 🌍 **SDG 2 aligned** — Every prediction includes food security impact

---

## 📦 Prerequisites

- **Python 3.10+**
- **Docker** (for containerized deployment)
- **Groq API Key** — Get one free at [console.groq.com](https://console.groq.com)
- **Pre-trained model file** — Download `plant_disease_model_1_latest.pt` from the [Google Drive link](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link) and place it in the `Flask Deployed App/` directory.

---

## 🛠 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/SubbuVasanth/plant-disease-prediction.git
cd Plant-Disease-Detection/Flask\ Deployed\ App
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key

```bash
export GROQ_API_KEY="gsk_your_api_key_here"     # Linux/Mac
# set GROQ_API_KEY=gsk_your_api_key_here        # Windows CMD
# $env:GROQ_API_KEY="gsk_your_api_key_here"     # Windows PowerShell
```

### 5. Download the model file

Download `plant_disease_model_1_latest.pt` from the [Google Drive link](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link) and place it in this directory.

---

## ▶️ Run Locally

```bash
cd "Flask Deployed App"
python app.py
```

The app will start at **http://localhost:5000**

- **Web UI**: Open http://localhost:5000 in your browser
- **AI Engine**: Navigate to http://localhost:5000/index
- **API**: Send POST requests to http://localhost:5000/predict

---

## 🐳 Run with Docker

### Option 1: Using the Shell Script

```bash
export GROQ_API_KEY="gsk_your_api_key_here"
cd "Flask Deployed App"
bash build_run.sh
```

### Option 2: Manual Docker Commands

```bash
# Build the image
cd "Flask Deployed App"
docker build -t agriscan-ai .

# Run the container
docker run -d \
  --name agriscan-ai-container \
  -p 5000:5000 \
  -e GROQ_API_KEY="gsk_your_api_key_here" \
  agriscan-ai
```

### Option 3: Pull from DockerHub

```bash
docker pull subbuvasanthk/agriscan-ai:latest

docker run -d \
  --name agriscan-ai-container \
  -p 5000:5000 \
  -e GROQ_API_KEY="gsk_your_api_key_here" \
  subbuvasanthk/agriscan-ai:latest
```

---

## 📡 API Usage

### `POST /predict`

Upload a plant leaf image and receive CNN + LLM combined analysis.

#### Sample `curl` command

```bash
curl -X POST -F "file=@test_images/AppleScab1.JPG" http://localhost:5000/predict
```

#### Sample Response

```json
{
  "cnn_prediction": {
    "class": "Apple___Apple_scab",
    "class_index": 0,
    "confidence": 0.9847
  },
  "llm_diagnosis": {
    "disease_name": "Apple Scab",
    "diagnosis": "Apple scab is a fungal disease caused by Venturia inaequalis that affects apple trees, causing dark olive-green to black lesions on leaves, fruit, and sometimes twigs. The CNN model has identified this condition with 98.47% confidence.",
    "treatment_steps": [
      "Apply fungicide sprays containing captan or myclobutanil during early spring.",
      "Remove and destroy infected fallen leaves to reduce fungal spore sources.",
      "Prune affected branches to improve air circulation within the tree canopy."
    ],
    "prevention_tips": [
      "Plant scab-resistant apple varieties such as Liberty, Enterprise, or Pristine.",
      "Maintain proper tree spacing to ensure adequate air circulation.",
      "Apply preventive fungicide sprays before rainy periods in spring."
    ],
    "sdg2_impact": "Early detection of apple scab supports UN SDG 2 (Zero Hunger) by preventing up to 70% yield loss in untreated orchards, ensuring stable apple production for food security, and reducing the need for excessive pesticide use — promoting sustainable agricultural practices."
  }
}
```

### Other Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Home page (Web UI) |
| `/index` | GET | AI Engine — upload leaf image (Web UI) |
| `/submit` | POST | Submit image for analysis (Web UI) |
| `/market` | GET | Supplement/fertilizer marketplace |
| `/contact` | GET | Contact us page |
| `/predict` | POST | **REST API** — JSON response (CNN + LLM) |

---

## 🐳 DockerHub

### Pull the image

```bash
docker pull subbuvasanthk/agriscan-ai:latest
```

### Push commands (for maintainers)

```bash
docker login
docker tag agriscan-ai subbuvasanthk/agriscan-ai:latest
docker push subbuvasanthk/agriscan-ai:latest
```

---

## 📁 Project Structure

```
Plant-Disease-Detection/
├── Flask Deployed App/
│   ├── app.py                  # Flask application (CNN + Groq integration)
│   ├── CNN.py                  # PyTorch CNN model definition (39-class)
│   ├── groq_agent.py           # Groq LLM agent for diagnosis generation
│   ├── Dockerfile              # Container configuration
│   ├── build_run.sh            # Build & run shell script
│   ├── requirements.txt        # Python dependencies (includes groq)
│   ├── disease_info.csv        # Disease information database
│   ├── supplement_info.csv     # Supplement/fertilizer database
│   ├── plant_disease_model_1_latest.pt  # Pre-trained CNN weights (download separately)
│   ├── templates/              # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── index.html
│   │   ├── submit.html
│   │   ├── market.html
│   │   └── contact-us.html
│   └── static/
│       └── uploads/            # Uploaded leaf images
├── Model/
│   ├── Plant Disease Detection Code.ipynb
│   └── model.JPG
├── test_images/                # Sample test images
├── demo_images/                # Screenshots of the web app
└── README.md                   # This file
```

---

## 🧪 Tech Stack

| Component | Technology |
|---|---|
| **Deep Learning Framework** | PyTorch 1.8 |
| **CNN Architecture** | Custom 4-block Conv2D + BatchNorm + MaxPool |
| **LLM Provider** | Groq Cloud (LLaMA 3.3-70B Versatile) |
| **Web Framework** | Flask 1.1.2 |
| **Inference Engine** | Flask REST API |
| **Containerization** | Docker (python:3.10-slim) |
| **Dataset** | PlantVillage (39 classes, 14 species) |
| **SDG Alignment** | UN SDG 2 — Zero Hunger |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is open source and available for educational and research purposes. Original CNN model and Flask app by [manthan89-py](https://github.com/manthan89-py).

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** — For providing the training data
- **Groq** — For ultra-fast LLM inference
- **manthan89-py** — For the original Plant Disease Detection project
- **United Nations** — SDG 2 Zero Hunger framework

---

<p align="center">
  Made with 💚 for sustainable agriculture and food security<br>
  <strong>UN SDG 2 — Zero Hunger</strong>
</p>
