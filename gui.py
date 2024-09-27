import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
from PIL import Image, ImageTk
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import sys

model = tf_keras.models.load_model('food_pred.h5', custom_objects={'KerasLayer': hub.KerasLayer})

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue") 

class FoodRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Food Recognition")
        self.geometry("600x600")  

        if sys.platform.startswith('win'):
            self.iconbitmap("appIcon.ico")  
        else:
            img = Image.open("appIcon.png")
            self.iconphoto(True, ImageTk.PhotoImage(img))


        self.heading_label = ctk.CTkLabel(self, text="Food Recognition", font=("Helvetica", 24))
        self.heading_label.pack(pady=20)

        self.upload_button = ctk.CTkButton(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = ctk.CTkLabel(self, text="No Image Uploaded", width=400, height=300)
        self.image_label.pack(pady=10)

        self.prediction_label = ctk.CTkLabel(self, text="Prediction: None", font=("Helvetica", 18))
        self.prediction_label.pack(pady=20)

        self.footer_label = ctk.CTkLabel(self, text="Supported Formats: .jpg, .jpeg, .png, .gif, .bmp", font=("Helvetica", 12))
        self.footer_label.pack(pady=10)

        self.image = None 

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
        )
        
        if not file_path:
            return

        if not self.is_valid_image(file_path):
            messagebox.showerror("Invalid file", "Please upload a valid image file.")
            return

        self.image = Image.open(file_path).convert('RGB')
        self.display_image(self.image)

        prediction = self.get_prediction(self.image)
        self.prediction_label.configure(text=f"Prediction: {prediction}")

    def display_image(self, img):
        display_img = img.resize((400, 300))  
        img_tk = ImageTk.PhotoImage(display_img)

        self.image_label.configure(image=img_tk, text="")
        self.image_label.image = img_tk 

    def is_valid_image(self, file_path):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        _, ext = os.path.splitext(file_path)
        return ext.lower() in valid_extensions

    def get_prediction(self, image):
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0 
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)

        class_names = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']
        return class_names[predicted_class[0]]


if __name__ == "__main__":
    app = FoodRecognitionApp()
    app.mainloop()
