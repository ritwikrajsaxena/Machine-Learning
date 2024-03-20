import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import cv2 as cv
import os
import numpy as np

from pathlib import Path
from skimage.io import imread
from skimage.transform import resize

from PIL import Image

from sklearn import svm

from keras.preprocessing import image
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread

from tensorflow.keras.preprocessing import image as keras_image
import requests
from io import BytesIO

print("Files imported successfully")
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
pd.set_option('display.max_columns', 30*30 + 1)

main_directory = "C:/Users/Raj/Desktop/pokemon-image-recognition/Small Test Set"
train_dir = "C:/Users/Raj/Desktop/pokemon-image-recognition/Small Test Set"
test_dir = "C:/Users/Raj/Desktop/pokemon-image-recognition/Small Test Set"

# Initialize a count for the number of directories
directory_count = 0

# List the contents of the directory
contents = os.listdir(main_directory)

# Iterate through the contents and count directories
for item in contents:
    item_path = os.path.join(main_directory, item)
    if os.path.isdir(item_path):
        directory_count += 1

print("Number of directories:", directory_count)



# Set model parameters
num_classes = directory_count  # Adjust the number of classes according to your Pokémon dataset
img_height, img_width = 75, 75  # Adjust image dimensions as needed
batch_size = 32

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

train_labels = train_generator.class_indices
test_labels = test_generator.class_indices

train_labels_list = []
for label, index in train_labels.items():
    train_labels_list.append(index)

# Convert list of indices to a numpy array
train_labels_array = np.array(train_labels_list)

test_labels_list = []
for label, index in test_labels.items():
    test_labels_list.append(index)

# Convert list of indices to a numpy array
test_labels_array = np.array(test_labels_list)

# Build Inception-ResNet model
base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))

train_features = base_model.predict_generator(train_generator)
test_features = base_model.predict_generator(test_generator)

# Apply GlobalAveragePooling2D to reduce dimensions
train_features = GlobalAveragePooling2D()(train_features)
test_features = GlobalAveragePooling2D()(test_features)

# Add a Dense layer for further feature transformation
train_features = Dense(1024, activation='relu')(train_features)
test_features = Dense(1024, activation='relu')(test_features)

# Reshape the features for SVM input if needed
train_features = np.reshape(train_features, (train_features.shape[0], -1))
test_features = np.reshape(test_features, (test_features.shape[0], -1))

min_samples = min(train_features.shape[0], train_labels_array.shape[0])
train_features_trimmed = train_features[:min_samples]
train_labels_array_trimmed = train_labels_array[:min_samples]

min_samples_test = min(test_features.shape[0], test_labels_array.shape[0])
test_features_trimmed = test_features[:min_samples_test]
test_labels_array_trimmed = test_labels_array[:min_samples]

# Initialize and train SVM classifier
svm_classifier = svm.SVC(kernel='linear')  # You can choose different kernels based on your data
svm_classifier.fit(train_features_trimmed, train_labels_array_trimmed)

# Evaluate SVM classifier
accuracy = svm_classifier.score(test_features_trimmed, test_labels_array_trimmed)
print("Accuracy of SVM classifier:", accuracy)


"""
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)
"""

# Function to preprocess user-provided image
def preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=(img_height, img_width))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image data
    return img_array

# Function to predict class and print result for user-provided image
def predict_pokemon_class(image_path):
    img_array = preprocess_image(image_path)
    features = base_model.predict(img_array)
    features = GlobalAveragePooling2D()(features)
    features = Dense(1024, activation='relu')(features)
    features = np.reshape(features, (1, -1))  # Reshape for SVM input

    predicted_class_index = svm_classifier.predict(features)[0]  # Get predicted class index
    predicted_probabilities = svm_classifier.predict_proba(features)  # Get predicted probabilities

    class_names = sorted(train_generator.class_indices, key=train_generator.class_indices.get)
    if 0 <= predicted_class_index < len(class_names):
        predicted_class_name = class_names[predicted_class_index]
        print("Predicted Pokemon Class Index:", predicted_class_index)
        print("Predicted Pokemon Class Name:", predicted_class_name)
        print("Predicted Probability:", predicted_probabilities[0, predicted_class_index])
    else:
        print("The model predicts that the image you entered does not belong to any of the classes of Pokemon it has the capability to identify.")

# Example: Input image path (Replace this with the path to your user-provided image)
user_image_path = input("Enter the path to the Pokemon image you want to be predicted: ")

# Predict class for user-provided image
predict_pokemon_class(user_image_path)