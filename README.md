# Vitamin Deficiency Detection Using Image Analysis

This project presents a non-invasive, AI-powered framework to detect vitamin deficiencies using image analysis. By leveraging deep learning and computer vision, the system examines visual indicators from the eyes, lips, tongue, and nails to identify deficiencies in Vitamins A, B, C, D, and E. The goal is to provide a scalable, cost-effective, and accessible diagnostic tool, especially for resource-limited environments.

---

## 📌 Features

- ✅ Image-based detection of vitamin deficiencies (multi-deficiency support)
- 🤖 Deep learning models with transfer learning (EfficientNet, VGGNet)
- 🔬 Advanced preprocessing and feature extraction
- 📊 Model explainability using Grad-CAM visualizations
- 🥗 Personalized dietary and lifestyle recommendations
- 🌐 Flask-based web application for user interaction

---

## 🛠️ Technologies Used

- Python, OpenCV, NumPy, Pandas
- TensorFlow / Keras (EfficientNet, VGGNet)
- Flask (Web Application Framework)
- Grad-CAM for model interpretability
- Matplotlib, Seaborn (Visualizations)

---


## 🔍 How It Works

1. User uploads an image (eye, lip, tongue, or nail).
2. Image is preprocessed and passed through the deep learning model.
3. System classifies vitamin deficiencies (supports multi-label).
4. Recommendations are generated based on the deficiencies.
5. Results and Grad-CAM visualizations are displayed in the web app.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip or conda
