# Wheat spike phenotyping and number of spikelets prediction 🌾

This repository contains scripts and a pre-trained model to process wheat spike images, extract phenotypic metrics (such as area, width, and length), and predict the number of spikelets. These tools are based on image processing techniques using OpenCV and a machine learning pipeline trained on Support Vector Regression (SVR).

## 📂 Repository Structure

wheat-phenotyping-model/
├── Entrenamiento_nuevo.py          # Script to train a new SVR model using spike metrics
├── procesamiento_img.py            # Script to process images and extract phenotypic metrics
├── procesamiento_con_prediccion.py # Script to process images and predict the number of spikelets
├── model_NUEVO_SVR.joblib          # Pre-trained SVR model for spikelet prediction
├── requirements.txt                # Dependencies for running the project
├── README.md                       # Documentation


## 🚀 Features

1. **Image Processing:**
   - Removes awns from spikes using OpenCV's morphological operations.
   - Extracts phenotypic metrics such as area, width, and length of the spikes.

2. **Machine Learning Model:**
   - Predicts the number of spikelets based on extracted metrics using a pre-trained SVR model.

3. **Scalability:**
   - Processes and predicts metrics for multiple images in batches, making it suitable for high-throughput phenotyping.

---
🧑‍💻 How to Use the Repository
1. Train a New Model
If you want to retrain the SVR model using your own data:

Ensure your dataset (e.g., metricas_nuevas_2.xlsx) has the columns: largo, ancho, area, and espiguillas.
Update the file path in Entrenamiento_nuevo.py

