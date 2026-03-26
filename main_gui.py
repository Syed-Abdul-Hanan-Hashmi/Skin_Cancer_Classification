import os
import sys

# --- 1. ENVIRONMENT STABILITY ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import gc
import tensorflow as tf

class SkinCancerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Cancer Diagnostic Suite - FYP Final")
        self.root.geometry("800x850")
        self.root.configure(bg="#1e1e1e")

        self.base_path = r"C:\Users\Hanan Hashmi\Documents\github\Skin_Cancer_Classification\models"
        self.classes = ["Benign", "Benign Keratosis", "Malignant"]
        
        self.model_files = {
            "Custom CNN (Regularized)": "skin_model_custom_regularized_v1.h5",
            "DenseNet121": "skin_model_densenet121_v1.h5",
            "MobileNetV2": "skin_model_final_v2.h5",
            "ResNet50": "skin_model_resnet50_v1.h5",
            "ResNet50V2": "skin_model_resnet50v2_v1.h5"
        }

        self.current_model = None
        self.current_model_name = None
        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="SKIN LESION ANALYSIS SYSTEM", 
                 font=("Arial", 22, "bold"), fg="#00d1b2", bg="#1e1e1e").pack(pady=30)
        
        ctrl = tk.Frame(self.root, bg="#1e1e1e")
        ctrl.pack(pady=10)

        self.model_selector = ttk.Combobox(ctrl, values=list(self.model_files.keys()), state="readonly", width=35)
        self.model_selector.current(0)
        self.model_selector.pack(side="left", padx=10)

        tk.Button(ctrl, text="UPLOAD & ANALYZE", command=self.run_analysis, 
                  bg="#007acc", fg="white", font=("Arial", 10, "bold"), padx=25).pack(side="left")

        self.lbl_main = tk.Label(self.root, bg="#252526", width=450, height=450)
        self.lbl_main.pack(pady=30)

        self.res_lbl = tk.Label(self.root, text="System Ready", font=("Arial", 18, "bold"), 
                                fg="#ecf0f1", bg="#1e1e1e")
        self.res_lbl.pack(pady=20)

    def run_analysis(self):
        img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not img_path: return

        selected = self.model_selector.get()
        self.res_lbl.config(text=f"Analyzing with {selected}...", fg="#f1c40f")
        self.root.update()

        try:
            # Memory Wipe to prevent frozen states
            tf.keras.backend.clear_session()
            gc.collect()

            # Load Model
            if self.current_model_name != selected:
                full_model_path = os.path.join(self.base_path, self.model_files[selected])
                self.current_model = tf.keras.models.load_model(full_model_path, compile=False, safe_mode=False)
                self.current_model_name = selected

            # Read and Resize Image
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            # --- THE UNIVERSAL FIX ---
            # All models expect raw 0-255 pixels in a float32 array
            processed_input = np.expand_dims(img_resized.astype('float32'), axis=0)

            # Predict
            preds = self.current_model.predict(processed_input, verbose=0)
            
            self.show_results(img_resized, preds)

        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")

    def show_results(self, img_resized, preds):
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_resized).resize((450, 450)))
        self.lbl_main.config(image=img_tk)
        self.lbl_main.image = img_tk 

        idx = np.argmax(preds[0])
        conf = preds[0][idx] * 100
        
        res_color = "#e74c3c" if self.classes[idx] == "Malignant" else "#00d1b2"
        self.res_lbl.config(text=f"Result: {self.classes[idx]} ({conf:.2f}%)", fg=res_color)

if __name__ == "__main__":
    root = tk.Tk()
    app = SkinCancerGUI(root)
    root.mainloop()