import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_merge_data(labels_file, metadata_file):
    try:
        labels = pd.read_csv(labels_file)
        info = pd.read_csv(metadata_file).drop('image', axis=1)
        return pd.concat([labels, info], axis=1)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def clean_data(df, target_column='MEL'):
    df['lesion_id'].fillna(method='bfill', inplace=True)
    df['age_approx'].fillna(df['age_approx'].median(), inplace=True)
    df['sex'].fillna(df['sex'].mode()[0], inplace=True)
    df['anatom_site_general'].fillna(method='ffill', inplace=True)

    # Encode categorical variables
    anatom_site_general = {'anterior torso': 1, 'upper extremity': 2, 'posterior torso': 3, 'lower extremity': 4,
                           'lateral torso': 5, 'head/neck': 6, 'palms/soles': 7, 'oral/genital': 8}
    sex = {'male': 0, 'female': 1}
    df['anatom_site_general'] = df['anatom_site_general'].map(anatom_site_general)
    df['sex'] = df['sex'].map(sex)

    # Extract target and features
    target = df[[target_column]].values
    features = df.drop(['image', target_column, 'lesion_id'] + [col for col in df.columns if col.startswith(('MEL', 'NV')) and col != target_column], axis=1)
    return features, target

def load_images(image_dir, img_size=(32, 32)):
    images = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img_size)
                images.append(img)
        except Exception as e:
            logging.warning(f"Error loading image {img_path}: {e}")
    return images

def prepare_data(features, target, images, train_size=0.8, val_size=0.1, output_dir="output"):

    if len(images) != len(features):
        logging.error("Mismatch between number of images and features")
        return None

    # Split data
    train_idx, temp_idx = train_test_split(range(len(images)), train_size=train_size, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_size / (1 - train_size), random_state=42)

    train_images = np.array([images[i] for i in train_idx], dtype='float32')
    val_images = np.array([images[i] for i in val_idx], dtype='float32')
    test_images = np.array([images[i] for i in test_idx], dtype='float32')
    x_train = features.iloc[train_idx]
    x_val = features.iloc[val_idx]
    x_test = features.iloc[test_idx]
    y_train = target[train_idx]
    y_val = target[val_idx]
    y_test = target[test_idx]

    # Normalize images
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    val_images = (val_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # Save scaler
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    return train_images, val_images, test_images, x_train, x_val, x_test, y_train, y_val, y_test, scaler
