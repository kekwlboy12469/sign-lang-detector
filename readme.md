# 🤟 Real-Time Sign Language Detector

This project allows real-time sign language detection using a webcam and machine learning. It uses **MediaPipe** for hand landmark detection and an ML model trained on your custom signs.

---

## 📸 Features

- Real-time sign detection using webcam
- Train your own custom gestures
- Lightweight and fast
- Easy to run locally

---

## 📁 Project Structure

📦SIGN-LANGUAGE-PROJECT
├── models/ # Saved model, scaler, label encoder
├── sample/ # Sample sign gesture images
├── sign_data/ # Saved landmark data per gesture
├── live.py # MAIN app - real-time sign detection
├── webcam.py #optional - for testing
├── train_model.py # Train model using collected data
├── prepare_dataset.py # Extract and save hand landmarks
├── utils.py # to save dataset save custom sign lang
├── requirements.txt # All dependencies


---

## 🛠️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/kekwlboy12469sign-language-detector.git
cd sign-language-detector
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate
pip install -r requirements.txt

🔐 Note
This app runs locally only due to browser restrictions on webcam access in Streamlit Cloud. You can fork/clone it and use it freely!

📣 Contributions
PRs welcome. You can contribute by:

Improving UI

Supporting dynamic model retraining

Adding gesture audio feedback
