# 🧠 Brain Tumor Detection Web App  
A deep learning–powered web application built with **Python**, **TensorFlow**, and **Streamlit** to detect brain tumors from MRI images.

🔴 **Live Demo Link:**  
https://brain-tumor-detection-5ta6rtfbao6eeajtljf69v.streamlit.app/

---

## 📖 About The Project  
This project is designed as a complete end-to-end **machine learning + web deployment** application.  
It showcases how a trained CNN model can be integrated into a clean, user-friendly web interface to make real-time predictions.

### 🎯 **Project Objectives**
- Build a deep learning model capable of detecting brain tumors from MRI images.
- Provide an interactive and simple UI for non-technical users.
- Deploy the app online so anyone can access it instantly.
- Demonstrate a complete ML lifecycle: training → testing → deployment.

---

## 📝 **Project Overview**

When a user uploads an MRI image (JPG/PNG), the app:

1. Preprocesses the image (resize, normalize).  
2. Feeds it into a trained CNN model (`.h5` file).  
3. Predicts **Tumor** / **No Tumor**.  
4. Displays:
   - The predicted class  
   - Confidence score  
   - Raw probabilities  
   - The MRI image preview  

The entire pipeline runs instantly inside the Streamlit interface.

---

## 🛠️ **Technologies Used**

| Component | Technology |
|----------|------------|
| Language | Python |
| ML / DL | TensorFlow, Keras |
| Web Framework | Streamlit |
| Image Processing | Pillow |
| Math & Data | NumPy |
| Deployment | Streamlit Cloud |

---

## 📂 **Project Structure**

app.py → Main Streamlit application file

BrainTumor10Epochscategorical.h5 → Trained CNN model

maintest.py → Script for testing the model locally

maintrain.py → Script used to train the model

requirements.txt → Python dependencies

README.md → Project documentation

.gitignore → Files and folders to ignore in Git


