
# 🎤 End-to-End Speech Emotion Recognition using ANN

This project explores the use of deep learning to recognize emotions from speech. By analyzing audio features like tone and pitch, the model learns to classify emotions such as happy, sad, angry, and neutral from voice recordings.

---

## 🎯 Objective

To build an Artificial Neural Network (ANN) and Multi-Layer Perceptron (MLP) model capable of detecting human emotions from speech audio clips using deep learning and machine learning techniques.

---

## 🧪 Business Use Case

Speech Emotion Recognition (SER) has broad applications:

- **Healthcare**: Supporting mental health by analyzing vocal emotion.
- **Customer Service**: Prioritizing and routing calls based on emotional tone.
- **Education**: Enhancing engagement by responding to student emotions.
- **Entertainment**: Creating more dynamic interactions in gaming and virtual assistants.

---

## 📁 Folder Structure

```
project_root/
├── input/              # Contains the RAVDESS audio dataset
├── output/             # Trained model files (Keras and Sklearn)
├── src/
│   ├── engine.py       # Entry point for training and inference
│   └── ml_pipeline/
│       ├── utils.py    # Feature extraction script
│       └── model.py    # Model building and evaluation logic
├── config.ini          # Model configuration parameters
├── requirements.txt    # Python package dependencies
```

---

## 📊 Dataset

- **Source**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Size**: 7356 `.wav` audio files from 24 actors
- **Emotions**: Neutral, calm, happy, sad, angry, fearful, disgust, and surprised
- **Used Files**: Only speech audio (not song clips) are used in this project

---

## ⚙️ Features Extracted

Using the `librosa` library, we extract the following features per audio clip:

- **MFCC**: Mel-frequency cepstral coefficients
- **Chroma**: Harmonic pitch content
- **Spectral Contrast**: Differences in peaks and valleys of sound spectrum

Each audio file is transformed into a 193-dimensional feature vector.

---

## 🧠 Model Architecture

### ANN (Keras + TensorFlow)
- Hidden layers: 300, 50, 20, 10
- Activation: ReLU (hidden), Softmax (output)
- Optimizer: Adam
- Loss: Categorical cross-entropy
- Output: Saved `.h5` model in `output/`

### MLP (scikit-learn)
- Used for comparison
- Simpler architecture, trained on same preprocessed features
- Output: Saved `.pkl` model in `output/`

Model architecture and training settings are controlled via the `config.ini` file.

---

## 🧪 Preprocessing

- Standardization of features (mean = 0, std = 1)
- 70/30 train-test split
- Stratified sampling to address class imbalance

---

## 🔁 Workflow Overview

1. Load and structure audio data (`input/`)
2. Extract audio features via `utils.py`
3. Preprocess features and split data
4. Train models with `model.py`
5. Save models in `output/`
6. Make predictions with `engine.py`

---

## 💻 How to Run

### Step 1: Create a virtual environment

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Extract features
```bash
cd src
python ml_pipeline/utils.py
```

### Step 4: Train models
```bash
python ml_pipeline/model.py
```

### Step 5: Make predictions
```bash
# Keras model
python engine.py --framework=keras --infer --infer-file-path="path_to_audio.wav"

# Sklearn model
python engine.py --framework=sklearn --infer --infer-file-path="path_to_audio.wav"
```

Use `--framework` to choose the model backend.

---

## 📌 Notes

- All trained models are saved to the `output/` folder
- Configurations such as epochs, batch size, and model paths are managed via `config.ini`
- The modular structure allows easy switching between models and data sources

---

## 🧠 Key Takeaways

- SER can enhance human-computer interaction across many fields
- Deep learning effectively models emotional tone in speech
- Modular design allows flexibility in model experimentation
- Feature engineering plays a key role in audio-based ML tasks

---

## 📍 Conclusion

This project showcases how emotion can be decoded from speech using machine learning and deep learning. It highlights how artificial intelligence can move toward more empathetic and emotionally aware systems.
