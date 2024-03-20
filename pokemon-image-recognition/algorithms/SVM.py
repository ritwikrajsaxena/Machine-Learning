import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV


# Define data directory
data_dir = "C:/Users/Raj/Desktop/pokemon-image-recognition/Small Test Set"

# Function to load images and labels
def load_data(data_directory):
    images = []
    labels = []
    classes = sorted(os.listdir(data_directory))
    label_encoder = LabelEncoder()

    for idx, folder in enumerate(classes):
        class_path = os.path.join(data_directory, folder)
        image_count = 0  # Initialize image_count
        for file in os.listdir(class_path):
           if image_count < 25 and file.lower().endswith(('.jpg', '.png', '.webp', '.gif', '.jpeg')):  # Considering only 25 images per class with these extensions
                image_path = os.path.join(class_path, file)
                image = Image.open(image_path)
                image = image.resize((224, 224))  # Resize images to a common size
                image = np.array(image)
                if image.shape == (224, 224, 3):  # Ensure the image has the correct shape
                    images.append(image)
                    labels.append(folder)
                    image_count += 1  # Increment image_count after adding an image
    
    encoded_labels = label_encoder.fit_transform(labels)
    return np.array(images), encoded_labels

# Load images and labels
images, labels = load_data(data_dir)

# Flatten and reshape the images
num_images = images.shape[0]
images = images.reshape(num_images, -1)

# Split the dataset into train and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize and train SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(train_images, train_labels)

# Calibrate the classifier for probability estimation
calibrated_svm = CalibratedClassifierCV(svm, method='sigmoid', cv='prefit')
calibrated_svm.fit(train_images, train_labels)

predictions = svm.predict(test_images)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy of SVM on test data: {accuracy}")

# Function to make predictions on a user-provided image
def predict_pokemon_from_image(img_path, model):
    # Load and preprocess the user-provided image
    input_image = Image.open(img_path)
    input_image = input_image.resize((224, 224))  # Resize image to match the model's expected input size
    input_image = np.array(input_image)
    input_image = input_image.reshape(1, -1)  # Flatten the image
    input_image = input_image / 255.0  # Normalize pixel values if required
    input_image = input_image.reshape(1, 224, 224, 4)  # Adjust the shape based on the model's input shape
    input_image_flat = input_image.flatten().reshape(1, -1)
    # Make predictions using the trained model
    predicted_class_index = model.predict(input_image_flat)[0]

    # Get the predicted class dynamically from the directory structure
    class_names = sorted(os.listdir(data_dir))
    if 0 <= predicted_class_index < len(class_names):
        predicted_class_name = class_names[predicted_class_index]
        print("Predicted Pokemon Class Index:", predicted_class_index)
        print("Predicted Pokemon Class Name:", predicted_class_name)

        # Get predicted probabilities
        predicted_probabilities = model.predict_proba(input_image_flat)
        print("Predicted Probabilities:", predicted_probabilities)
    else:
        print("The model predicts that the image you entered does not belong to any of the classes of Pokemon it has the capability to identify.")

# Get user input for the image path
user_image_path = input("Enter the path to the Pokemon image you want the model to make a prediction on: ")

# Make predictions on the user-provided image
predict_pokemon_from_image(user_image_path, calibrated_svm)