# 🔬 **Skin Cancer Diagnostic Suite (FYP)**

**An AI-powered medical imaging tool developed for a Final Year Project (FYP). This application utilizes deep learning to classify skin lesions with high precision, comparing five distinct architectures.**

---

## **📊 Dataset Specifications**
**The models were trained on a high-quality, balanced dataset to ensure diagnostic reliability across all categories.**

* **Total Dataset Size:** **30,000 Images**
* **Classes:** **3 (Benign, Benign Keratosis, Malignant)**
* **Class Distribution:** **Balanced (10,000 images per class)**
* **Data Split:** **80% Training (24,000), 10% Validation (3,000), 10% Testing (3,000)**

---

## **📥 Model Downloads**
**Due to GitHub's 100MB file size limit, the trained weights (.h5 files) are hosted externally. You must download the models and place them in a folder named `models/` in the project root to run the GUI.**

* 📂 **Download Trained Models:** [**Google Drive Link**](https://drive.google.com/drive/folders/14Pj_NR23ZLMvddsYngSfWzTXCEK7BoM4?usp=sharing)

---

## **🖥️ UI Features**
**The graphical user interface is optimized for single-view analysis and research efficiency.**

* **Multi-Model Toggle:** **Switch between 5 different CNN architectures in real-time.**
* **Unified Preprocessing:** **Handles raw pixel data (0-255) as required by the internal model rescaling layers.**
* **Memory Optimization:** **Uses `tf.keras.backend.clear_session` to prevent "frozen" predictions and memory leaks.**
* **Visual Feedback:** **Results are color-coded (Red for Malignant, Green for Benign) for instant status recognition.**

---

## **🧠 Supported Architectures**
1. **Custom CNN (Regularized):** **Features L2=0.001 regularization and Spatial Dropout.**
2. **DenseNet121:** **Leverages dense connectivity for efficient feature reuse.**
3. **MobileNetV2:** **Optimized for speed and mobile-level efficiency.**
4. **ResNet50:** **Utilizes residual learning to solve the vanishing gradient problem.**
5. **ResNet50V2:** **An improved version of the original ResNet architecture.**

---

## **🛠️ Setup & Installation**

### **1. Clone the Repository**
```bash
git clone [https://github.com/Syed-Abdul-Hanan-Hashmi/Skin_Cancer_Classification.git](https://github.com/Syed-Abdul-Hanan-Hashmi/Skin_Cancer_Classification.git)
cd Skin_Cancer_Classification