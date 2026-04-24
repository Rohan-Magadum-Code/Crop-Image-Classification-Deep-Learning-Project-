# 🌾 Crop Image Classification using Deep Learning

A **Deep Learning-based Crop Classification System** that predicts the type of crop from an uploaded image using a fine-tuned **ResNet18 model**. The project is deployed with an interactive **Streamlit web app** for real-time predictions.

---

## 🚀 Features

* 📷 Upload crop images and get instant predictions
* 🧠 Built using Transfer Learning (**ResNet18**)
* ⚡ Fast and lightweight inference on CPU
* 🌐 Interactive UI using Streamlit
* 📊 Trained on a large multi-class crop dataset

---

## 📂 Dataset

The model is trained on the **140 Most Popular Crops Image Dataset** from Kaggle.

🔗 Dataset Link:
https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset?resource=download-directory&select=RGB_224x224

### Dataset Details:

* 📁 140 crop classes
* 🖼️ Images resized to **224x224**
* 📊 Train / Validation / Test split available

---

## 🧠 Model Architecture

* Model: **ResNet18 (Transfer Learning)**
* Pretrained on ImageNet
* Final Fully Connected Layer modified for **140 classes**
* Loss Function: CrossEntropyLoss
* Optimizer: Adam

---

## 📈 Performance

* ✅ Validation Accuracy: ~47%
* ✅ Test Accuracy: ~46%

> ⚠️ Note: Accuracy can be improved with more training, data augmentation, and fine-tuning.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* Streamlit
* Joblib

---

## 🖥️ Streamlit App

The application allows users to upload an image and get predictions instantly.

### 📸 App Screenshot

<img width="1750" height="2479" alt="Streamlit Page Screenshot" src="https://github.com/user-attachments/assets/c22e378c-127e-4479-9a87-c35a447cd2cf" />

---

## 🔍 How It Works

1. User uploads an image
2. Image is preprocessed (resize, normalize)
3. Passed through trained ResNet18 model
4. Model predicts crop class
5. Result displayed on UI

---

## ⚠️ Known Issues

* Model accuracy is moderate (~46%)
* Large dataset may require high storage
* Model trained on GPU but deployed on CPU

---

## 🚀 Future Improvements

* 🔥 Improve accuracy using fine-tuning
* 📊 Add confidence score display
* 🌐 Deploy on cloud (Streamlit Cloud / AWS)
* 🧠 Try advanced models (EfficientNet, ViT)
* 📱 Mobile-friendly UI

---

## 👨‍💻 Author

**Rohan Magadum**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!

---
