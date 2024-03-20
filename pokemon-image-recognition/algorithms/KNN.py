# This is our "from scratch" algorithm. First, starting with imports:
from PIL import Image
from collections import Counter
import numpy as np
import os
import ast      # Used for the testing function.
import random   # Used for the testing function.

# For creating the statistics:
import matplotlib.pyplot as plt
import pandas as pd
import time

# Generates the Euclidean Distance between two vectors. These vectors are 256 in length (for the pixels).
def EucDisHistograms(H1,H2):
    distance = 0
    for i in range(len(H1)):
        distance += np.square(H1[i]-H2[i])
    return np.sqrt(distance)

def generate_training_data_with_export(export=True):
    # Import all the data as a 3D array based on colors:
    training_labels = []
    training_data = []

    # Get all of the images in the training set, then convert them to a 3D vector based off of color, then flatten them and add them to an array.
    # Create another list of the labels for each image, then zip them together so that we have parallel arrays.
    for root, dirs, files in os.walk("../../Small Test Set/", topdown=False):
        for name in dirs:
            for filename in os.listdir(os.path.join(root, name)):
                # Get the label of the Pokemon:
                training_labels.append(name)
                # Open the image and convert to an array:
                image = np.asarray(Image.open(os.path.join(root, name, filename)))
                # Flatten the image to turn it into a single dimension:
                training_value = image.flatten()
                # Create a histogram where we get the number of times that a pixel appears in the images:
                RH1 = Counter(training_value)

                # Get the counts of all the pixels, put it into the array, then append it to the training data.
                H1 = []
                for i in range(256):
                    if i in RH1.keys():
                        H1.append(RH1[i])
                    else:
                        H1.append(0)

                training_data.append(H1)
        
    # zipped_array = zip(training_labels, training_data)
    # print(tuple(zipped_array))

    # for i in range(0, len(training_labels)):
    #     print(training_labels[i] + "," + str(training_data[i]) + ",\n")

    if export:
        file = open("knn-data.txt", "w")
        for i in range(len(training_labels)):
            file.write(training_labels[i] + "," + str(training_data[i]) + "\n")

        file.close()

    print("Test is complete. knn-data.txt is generated.")

def run_test(k):
    training_data = []
    training_labels = []
    data_path = ""

    # Used for timing the algorithm:
    start = time.process_time()

    # These are used when running the analytics function.
    # Small Test Set (6 Pokemon):
    if (str(k) == 'a'):
        data_path = "Models/knn-data-1.txt"
    elif (str(k) == 'b'):
        data_path = "Models/knn-data-2.txt"
    elif (str(k) == 'c'):
        data_path = "Models/knn-data-3.txt"
    else:
        # Case when this is a number. This is when we're NOT getting the analytics.
        # Default to test on generation 1 dataset:
        data_path = "Models/knn-data-3.txt"

    # When running analytics, automatically switch k to be 4 for the following tests.
    if (type(k) is str):
        k = 4

    print("\nA test is being ran. Searching for knn-data.txt...")
    if (os.path.isfile(data_path)):
        print("File exists. Extracting the data.")
        with open(data_path) as f:
            data = f.readlines()
            for line in data:
                splittedData = line.split(',', 1)
                # Appending the label of the Pokemon.
                training_labels.append(splittedData[0])
                # Converting a string literal into a list. Appending it.
                training_data.append(ast.literal_eval(splittedData[1]))

    else:
        print("File doesn't exist. Please select another option.\n")
        main()

    # Get the number of elements that are in the set:
    numOfTestSamples = len(training_data)

    # Zip the two arrays together so that they stay linked. Cast to list. Then, randomize them for testing.
    training_set = list(zip(training_labels, training_data))
    random.shuffle(training_set)

    # Take a fourth of the list and split the list into two separate lists: Assign testing_set first so that training_set can be mutated:
    # If an error occurs here, then you need a bigger test set.
    testing_set = training_set[:numOfTestSamples // 4]
    # training_set = training_set[numOfTestSamples // 4:]

    # For computing the accuracy:
    numerator = 0
    denominator = len(testing_set)

    # For each of the elements in the testing set...
    for tests in testing_set:
        # Create an array that we will store the k-number of closest Euclidean distance(s).
        nearestNeighbor = []
        # Go through all the elements in the training set and compute the Euclidean Distance between the test/train sets:
        for trains in training_set:
            result = EucDisHistograms(tests[1], trains[1])
            # print("Nearest Neighbor: " + str(nearestNeighbor))
            if (len(nearestNeighbor) < k):
                nearestNeighbor.append([str(trains[0]), str(result)])

            else:
                for i in range(0, len(nearestNeighbor)):
                    # print("Comparing " + str(nearestNeighbor[i][1]) + " with " + str(result))
                    if ((float(nearestNeighbor[i][1])) > float(result)):
                        # Delete the value and replace it with the new result.
                        # print("Replacing!")
                        nearestNeighbor[i] = [str(trains[0]), str(result)]
                        # No need to traverse through the rest of the NN array. Go to the next element.
                        break

        # Nearest Neighbor is done. Calculate the best-approximated label:
        bestLabel = []
        for elements in nearestNeighbor:
            bestLabel.append(elements[0])

        # Create a "Counter" dictionary and get the first element, which is the most number of occurrences in the dictionary.
        prediction = next(iter(Counter(bestLabel)))

        # print("\nPrediction: " + prediction)
        # print("Actual answer: " + str(tests[0]) + "\n")

        if (prediction == str(tests[0])):
            numerator += 1

    # After the for loop, output the accuracy:
    print("Accuracy: %0.4f" % (numerator / denominator))

    return ([(numerator / denominator), str((time.process_time() - start))])

    # dist_test_ref_1 = EucDisHistograms(training_data[0], training_data[1])
    # print("The distance between %s and %s is : %0.4f" % (training_labels[0], training_labels[1], dist_test_ref_1))

    # dist_test_ref_2 = EucDisHistograms(training_data[0], training_data[2])
    # print("The distance between %s and %s is : %0.4f" % (training_labels[0], training_labels[2], dist_test_ref_2))

def main():
    print("This is the K-NN algorithm. Please make sure that the directories inside of the code is up-to-date.")
    print("Select an option:")
    print("1: ONLY create a training set - mainly used for debugging.")
    print("2: Run a test with existing data. This assumes you have ran Option 1 already.")
    print("3: Gather analytics (must have all three training models prepared)")
    userInput = int(input("Select an option: "))
    if (userInput == 1):
        print("Generate a data file to reuse for future tests? Type Y for yes (or leave blank) or N for no.\n")
        userInput = input()
        if (userInput == 'N'):
            generate_training_data_with_export(False)

        elif (userInput == 'Y' or userInput == ''):
            generate_training_data_with_export()

        else:
            print("Please enter valid input.\n")
            main()
        

    elif (userInput == 2):
        k = int(input("Select a k-value. Smaller will be a quicker computation: "))
        if isinstance(k, int):
            run_test(k)
        else:
            print("Error. Type in a number.\n\n")

    elif (userInput == 3):
        # This will automatically be tested with k = 4. If you want a different k-value, please
        # evaluate run_test and change the k-value in there for when k = 99.
        
        accuracy_times = []
        completion_times = []

        for code in range(ord('a'), ord('c') + 1):
            accuracy, process_time = run_test(chr(code))
            accuracy_times.append(accuracy)
            completion_times.append(process_time)

        print(accuracy_times)
        print(completion_times)

        completion_times = [eval(i) for i in completion_times]

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
        df['accuracy'].plot(ylabel='Accuracy (line)', kind='line', marker='d', ylim=(0, 1), secondary_y=True)
        # Set the x-ticks.
        ax1.set_xticklabels(["6 Pokemon", "151 Pokemon (Gen 1)", "721 Pokemon (Gen 1-6)"])
        # Create a title.
        plt.title("Accuracy and Computation Times for K-NN")
    
    else:
        print("Please select another option.\n")
        main()

main()