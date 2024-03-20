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

from keras.preprocessing import image
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
print("Files imported successfully")
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
pd.set_option('display.max_columns', 30*30 + 1)

# Define data directories
data_dir = "C:/Users/Raj/Desktop/pokemon-image-recognition/Small Test Set"
train_dir = "C:/Users/Raj/Desktop/pokemon-image-recognition/Small Test Set"
test_dir = "C:/Users/Raj/Desktop/pokemon-image-recognition/Small Test Set"

main_directory = "C:/Users/Raj/Desktop/pokemon-image-recognition/Small Test Set"

def preprocess_input_image(img_path):
    img = image.load_img(img_path, target_size=(75, 75))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    return img_array


def load_image_files(main_directory, dimension=(75, 75)):
    image_dir = Path(main_directory)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
   
    global images_list
    images_list = []
    # Get a list of subdirectories (each subdirectory represents an animal category)
    pokemon_categories = sorted(os.listdir(main_directory))

    # Initialize empty lists for features (X) and labels (y)
    X = []
    labels = []

    # Iterate through each animal category
    for category in pokemon_categories:
        category_path = os.path.join(main_directory, category)

        # Check if it's a directory
        if os.path.isdir(category_path):
            # List all image files in the category directory
            image_files = [f for f in os.listdir(category_path) if f.endswith(".jpg")]

            
            for image_file in image_files:
                image_path = os.path.join(category_path, image_file)

    for i, folder in enumerate(folders):
        # Get the label (pokemon name) for this folder
        label = folder.name
        
        # Load each image in the folder
        for file_path in folder.iterdir():


            img = preprocess_input_image(image_path)
               

            """
            # Read the image
            img = cv.imread(image_path)
            
            # Resize the image to the specified dimension
            
            img = resize(img, dimension, anti_aliasing=True)
            """

            # Append the image and its label to the lists

            images_list.append(img)
                        
            
            labels = np.append(labels, i)  # Use index as label
            
            # Convert lists to numpy arrays
            images = np.array(images_list)
            labels = np.array(labels)
    return images, labels
    

images, labels = load_image_files(main_directory)
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)




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

# Build Inception-ResNet model
base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
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

sample_images, sample_labels = train_generator.next()

# Print the first few images in the array
num_samples_to_print = 1  # Adjust the number of samples to print
for i in range(num_samples_to_print):
    print(f"Sample Image {i + 1}:")
    print(sample_images[i])
    print("Shape of the image:", sample_images[i].shape)
    print("=" * 50)


sample_images1, sample_labels1 = train_generator.next()
num_samples_to_print = 1  # Adjust the number of samples to print
for i in range(num_samples_to_print):
    print(f"Sample Image {i + 1}:")
    print(sample_images1[i])
    print("Shape of the image:", sample_images1[i].shape)
    print("=" * 50)

"""

"""
# Print model architecture and details
print("Model Summary:")
model.summary()

"""
"""
# Extract details about layers, filters, pooling, and activation functions
for layer in model.layers:
    print(f"Layer Name: {layer.name}")
    print(f"Stride: {layer.strides if hasattr(layer, 'strides') else 'N/A'}")
    print(f"Kernel Size: {layer.kernel_size if hasattr(layer, 'kernel_size') else 'N/A'}")
    print(f"Number of Kernels: {layer.filters if hasattr(layer, 'filters') else 'N/A'}")
    print(f"Number of Channels in Filter: {layer.input_shape[-1] if hasattr(layer, 'input_shape') else 'N/A'}")
    print(f"Padding: {layer.padding if hasattr(layer, 'padding') else 'N/A'}")
    print(f"Pooling: {layer.pooling if hasattr(layer, 'pooling') else 'N/A'}")
    print(f"Activation Function: {layer.activation if hasattr(layer, 'activation') else 'N/A'}")
    print(f"Number of Hidden Neurons: {layer.units if hasattr(layer, 'units') else 'N/A'}")
    
    
    if hasattr(layer, 'get_weights'):
        weights, biases = layer.get_weights()
        print(f"Weights Shape: {weights.shape if weights is not None else 'N/A'}")
        print(f"Biases Shape: {biases.shape if biases is not None else 'N/A'}")
    print("--------------")

# Total number of neurons in the model
total_neurons = sum(layer.count_params() for layer in model.layers)
print(f"Total Number of Neurons in the Model: {total_neurons}")


"""
test_data = images
test_labels = labels

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=2)

print("Test Accuracy:", test_accuracy)


def preprocess_input_image(img_path):
    img = image.load_img(img_path, target_size=(75, 75))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    return img_array

# Function to make predictions on a user-provided image
def predict_pokemon_from_image(img_path):
    # Preprocess the user-provided image
    input_image = preprocess_input_image(img_path)

    # Make predictions using the trained model
    predictions = model.predict(input_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)
    
    train_dir = "C:/Users/Raj/Desktop/pokemon-image-recognition/Small Test Set"
    if 0 < predicted_class_index <= num_classes and predictions[0, predicted_class_index] > 0:
        # Get the name of the predicted class dynamically from the directory structure
        class_names = sorted(os.listdir(train_dir))
        predicted_class_name = class_names[predicted_class_index]

        # Print the predicted class, its name, and the corresponding probability
        print("Predicted Pokemon Class Index:", predicted_class_index)
        print("Predicted Pokemon Class Name:", predicted_class_name)
        print("Predicted Probability:", predictions[0, predicted_class_index])
    else:
        # Print a message indicating that the model couldn't identify the Pokemon
        print("The model predicts that the image you entered does not belong to any of the classes of Pokemon it has the capability to identify.")

# Get user input for the image path
user_image_path = input("Enter the path to the Pokemon image: ")

# Make predictions on the user-provided image
predict_pokemon_from_image(user_image_path)