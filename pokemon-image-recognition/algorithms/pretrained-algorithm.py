from sentence_transformers import SentenceTransformer, util
import PIL.Image
from IPython.display import display
from IPython.display import Image
import torch
import os
import random
import csv
import time

# For creating the statistics:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_pretrained(user_input):
    # This model is used specifically for vectorizing images. Upon the first time running this command, it will download the model.
    knn_model_trainer = SentenceTransformer('clip-ViT-B-32')

    # Get the filenames and names of all the Pokemon:
    file_names = []
    labels = []

    if (user_input != 1):
        print("You selected a large dataset. Please be patient while this computes...")

    if (user_input == 1):
        for root, dirs, files in os.walk("../../Project Data/Small Test Set", topdown=False):
            for name in dirs:
                file_names.append(os.path.join(root, name))
                labels.append(name)

    elif (user_input == 2):
        for root, dirs, files in os.walk("../../Project Data/Gen 1 Set", topdown=False):
            for name in dirs:
                file_names.append(os.path.join(root, name))
                labels.append(name)

    elif (user_input == 3):
        for root, dirs, files in os.walk("../../Project Data/Complete Test Set", topdown=False):
            for name in dirs:
                file_names.append(os.path.join(root, name))
                labels.append(name)

    # Used for timing the algorithm:
    start = time.process_time()

    correctNames = labels
    random.shuffle(labels)

    img_names = []
    correct_labels = []

    for filepath in file_names:
        for filename in os.listdir(filepath):
            img_names.append((filepath + '/' + filename))
            correct_labels.append(filepath.split('/')[-1])

    # And compute the embeddings for these images
    img_emb = knn_model_trainer.encode(img_names)

    # And compute the text embeddings for these labels
    text_emb = knn_model_trainer.encode(labels)

    # Now, we compute the cosine similarity between the images and the labels
    cos_scores = util.cos_sim(img_emb, text_emb)

    # Then we look which label has the highest cosine similarity with the given images
    pred_labels = torch.argmax(cos_scores, dim=1)

    # Preparing the variables for statistics:
    predicted_correctly = 0
    overall_guessed = 0

    for img_name, pred_label, correct_labels in zip(img_names, pred_labels, correct_labels):
        if (user_input == 1):
            display(Image(img_name, width=200))
        print("Predicted label:", labels[pred_label])
        print("Correct label: " + correct_labels)
        print("\n\n")

        if (labels[pred_label] == correct_labels):
            predicted_correctly += 1

        overall_guessed += 1

    return ([(predicted_correctly / overall_guessed), (time.process_time() - start)])

def main():
    print("Select an number below.")
    print("1: Small Test Set (6 Pokemon)")
    print("2: First Generation (151 Pokemon)")
    print("3: Full Test Set (721 Pokemon)")
    print("4: Run all three results and provide statistics of each of them.")
    user_input = int(input("Pick a number: "))

    if (user_input >= 1 and user_input <= 3):
        accuracy, process_time = run_pretrained(user_input)
        print("Accuracy: " + str(accuracy))
        print("Computation time: %0.4f" % process_time)

    elif (user_input == 4):
        # Create the accuracy and completion time lists:
        accuracy_times = []
        completion_times = []
        for i in range(1,4):
            accuracy, process_time = run_pretrained(i)
            accuracy_times.append(accuracy)
            completion_times.append(process_time)

        # Create the dataframe.
        df = pd.DataFrame({'accuracy': accuracy_times,
                   'time': completion_times})
        # Change font size.
        plt.rcParams.update({'font.size': 18})
        # Create subplots to show two different plots.
        fig, ax1 = plt.subplots(figsize=(15, 10))
        # Create a bar plot for computation time.
        df['time'].plot(xlabel='Computation time in seconds (bar)', kind='bar', color='r')
        # Create a line plot for accuracy.
        df['accuracy'].plot(ylabel='Accuracy (line)', kind='line', marker='d', secondary_y=True)
        # Set the x-ticks.
        ax1.set_xticklabels(["6 Pokemon", "151 Pokemon (Gen 1)", "721 Pokemon (Gen 1-6)"])

main()