import tkinter as tk
import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Feed")

        # Create a canvas to display the camera feed
        self.canvas = tk.Canvas(root, width=250, height=300)
        self.canvas.pack(padx = 25)
        # Create verification button
        self.button = tk.Button(root, text="Verify", font=("Arial", 15), command=self.func)
        self.button.pack(pady = 20)
        # Create result label
        self.label = tk.Label(root, text="", width=20, font=("Arial", 15), bg="gray")
        self.label.pack(pady = 20)
        # Open the camera (usually camera index 0)
        self.cap = cv2.VideoCapture(0)

        # Create a function to update the camera feed on the canvas
        self.update()

    def func(self):
        self.label.config(text="testing")

        # Save input image to application_data/input_image folder
        cv2.imwrite(os.path.join("application_data", "input_image", "input_image.jpg"), self.frame)

    def update(self):
        # Read a frame from the camera
        ret, self.frame = self.cap.read()

        self.frame = self.frame[120:370, 200:450, :]

        if ret:
            # Convert the OpenCV frame to a PIL Image
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)

            # Update the canvas with the new frame
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.photo = photo

        # Schedule the next update
        self.root.after(10, self.update)

    def close(self):
        # Release the camera and close the Tkinter window
        self.cap.release()
        self.root.destroy()

    def preprocess(self, file_path):
        # Get byte code of image (the file path) and then decode it
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)

        img = tf.image.resize(img, (105, 105)) # Resizing out image according to the "Siamese Neural Networks"
                                            # research paper
        img = img / 255.0 # Scale every pixel value to 0-1 => scale the image
        return img

    def verify(self, model, detection_threshold, verification_threshold, app_ver_path):
        # Build results array
        results = []
        for image in os.listdir(os.path.join(app_ver_path)):
            input_img = self.preprocess(os.path.join("application_data", "input_image", "input_image.jpg"))
            validation_img = self.preprocess(os.path.join("application_data", "verification_images", image))

            # Make predictions
            result = model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))
            results.append(result)
        
        # Detection Threshold: A metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions in regards to the total positive samples
        verification = detection / len(os.listdir(os.path.join("application_data", "verification_images")))
        verified = verification > verification_threshold
        
        return results, verified


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()