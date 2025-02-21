import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_image(image, label, title="Image", output_dir="output/plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.imshow(image)
    plt.title(f"{title} - Label: {np.argmax(label)}")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()
    logging.info(f"Image saved to {output_dir}")

def predict_new_data(model, image_path, metadata, scaler, output_dir="output"):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at {image_path}")
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (32, 32))
        img = img.astype('float32')
        mean = np.mean(img, axis=(0, 1, 2))
        std = np.std(img, axis=(0, 1, 2))
        img = (img - mean) / (std + 1e-7)
        img = np.expand_dims(img, axis=0)

        metadata = np.array([metadata]).astype('float32')
        metadata = scaler.transform(metadata)

        prediction = model.predict([img, metadata])[0]
        class_idx = np.argmax(prediction)
        class_label = 'Melanoma' if class_idx == 1 else 'Non-Melanoma'
        logging.info(f"Prediction for {image_path}: {class_label} (Prob: {prediction[class_idx]:.4f})")
        return class_label
    except Exception as e:
        logging.error(f"Error predicting on new data: {e}")
        return None
