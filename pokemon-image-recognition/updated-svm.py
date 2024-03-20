
import pandas as pd 
import os 
from skimage.transform import resize 
from skimage.io import imread 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import pickle

# For analytics:
import time

def predict(path):
    Categories=[] 
    for root, dirs, files in os.walk("C:/Users/Eric/Desktop/Project/", topdown=False):
        for name in dirs:
            Categories.append(name)

    with open('svm-model-new.pkl', 'rb') as f:
        model = pickle.load(f)

        img=imread(path) 
        plt.imshow(img) 
        plt.show() 
        img_resize=resize(img,(150,150,3)) 
        l=[img_resize.flatten()] 
        probability=model.predict_proba(l) 
        for ind,val in enumerate(Categories): 
            print(f'{val} = {probability[0][ind]*100}%') 
        print("The predicted image is : "+Categories[model.predict(l)[0]])

def generateAnalytics(dataset):
    Categories=[] 

    start = time.process_time()

    for root, dirs, files in os.walk("C:/Users/Eric/Desktop/Project/" + dataset, topdown=False):
        for name in dirs:
            Categories.append(name)

    flat_data_arr=[] #input array 
    target_arr=[] #output array 
    datadir="C:/Users/Eric/Desktop/Project/" + dataset 
    #path which contains all the categories of images 
    for i in Categories: 
        # print(f'loading... category : {i}') 
        path=os.path.join(datadir,i) 
        for img in os.listdir(path): 
            img_array=imread(os.path.join(path,img)) 
            img_resized=resize(img_array,(150,150,3)) 
            element = img_resized.flatten()
            if (len(element) == 202500):
                element = element[:67500]
            # print(len(element))
            flat_data_arr.append(element) 
            target_arr.append(Categories.index(i)) 
        # print(f'loaded category:{i} successfully') 
    flat_data=np.array(flat_data_arr) 
    target=np.array(target_arr)

    #dataframe 
    df=pd.DataFrame(flat_data)
    df['Target']=target 
    df.shape

    #input data  
    x=df.iloc[:,:-1]  
    #output data 
    y=df.iloc[:,-1]

    print("Splitting")

    # Splitting the data into training and testing sets 
    x_train,x_test,y_train,y_test=train_test_split(x,y) 


    # Defining the parameters grid for GridSearchCV 
    param_grid={'C':[0.1,1,10,100], 
                'gamma':[0.0001,0.001,0.1,1], 
                'kernel':['rbf','poly']} 
    
    # Creating a support vector classifier 
    svc=svm.SVC(probability=True) 
    
    # Creating a model using GridSearchCV with the parameters grid 
    model=GridSearchCV(svc,param_grid)

    # Training the model using the training data 
    print("Fitting")
    model.fit(x_train,y_train)

    # Saving the model: 
    print("Saving the model. Please wait.")
    with open('svm-model-new.pkl','wb') as f:
        pickle.dump(model,f)

    # Testing the model using the testing data 
    print("Predicting")
    y_pred = model.predict(x_test) 
    
    # Calculating the accuracy of the model 
    accuracy = accuracy_score(y_pred, y_test) 
    
    # Print the accuracy of the model 
    print(f"The model is {accuracy*100}% accurate")

    return ([accuracy, (time.process_time() - start)])

def main():
    print("What would you like to do?")
    print("1: Predict a Pokémon")
    print("2: Run analytics (must be done before predicting a Pokémon - generates the model)")
    userInput = int(input())

    if (userInput == 1):
        print("Type the path that leads to the Pokémon that you want to predict:")
        path = str(input())
        predict(path)

    elif (userInput == 2):
        accuracy_times = []
        completion_times = []

        directories = ["Small Test Set"]
        for dataset in directories:
            accuracy, process_time = generateAnalytics(dataset)
            accuracy_times.append(accuracy)
            completion_times.append(process_time)

        print(accuracy_times)
        print(completion_times)

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
        plt.title("Accuracy and Computation Times for Support Vector Machines")

main()