import pandas as pd
import math
from copy import deepcopy


# function for calculating the accuracy of our classifier
def crossValidation(data, current_features, feature, add_or_remove):
    feature_set = current_features.copy()
    # add current feature if doing forward search and remove if doing backward elimination
    if (add_or_remove == "add"):
        feature_set.append(feature)
    else:
        feature_set.remove(feature)
    
    # features that will not be looked over can be set to 0 for better efficiency and accuracy
    data_copy = deepcopy(data)
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            if (j not in feature_set):
                data_copy[i][j] = 0
    
    # stores the number of correctly classified instances
    correctly_classified = 0
    
    # loop through entire dataset
    for i in range(len(data)):
        # make a list of objects in dataset to classify
        object_to_classify = data_copy[i][1:]       
        # save the label/classification of the previous list of objects (1 or 2)
        label_object_to_classify = data_copy[i][0]
        
        nearest_neighbor_distance = math.inf
        nearest_neighbor_location = math.inf
        
        # loop through entire dataset to determine euclidean distances and accuracy
        for k in range(len(data)):        
            # make sure not to compare the same feature to itself    
            if (k != i):
                # calculate euclidean distance of feature k with its nearest neighbors
                distance = sum([(a-b) ** 2 for a, b in zip(object_to_classify, data_copy[k][1:])])
                distance = math.sqrt(distance)
                
                # closer neighbor found to current feature so it must be saved
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data_copy[nearest_neighbor_location][0]
        
        # nearest neighbor is of the same label/classification so we can increment num of correctly classified
        if (label_object_to_classify == nearest_neighbor_label):
            correctly_classified += 1
        
    # return accuracy to search algorithm
    return (correctly_classified / len(data_copy))

    
# function for performing forward selection search on features in dataset 
def featureSearch(data):
    # create an empty feature set list; we will be adding features to this list iteratively
    current_features = []
    best_accuracy = 0
    
    # for loop to traverse down search tree
    for i in range(len(data)):
        feature_to_add = 0
        
        # for loop to traverse through each feature
        for j in range(1, len(data[i])):
            # consider adding new feature if it has not been considered already
            if (j not in current_features):
                # calculate accuracy of new feature by using cross validation function, making sure to add
                # new feature to the feature set list 
                accuracy = crossValidation(data, current_features, j, "add")
                # for printing current feature list only
                temp_list = current_features.copy()
                temp_list.append(j)
                percentage = "{:.2%}".format(accuracy)
                # print accuracy to user when attempting to use current set of features
                print("\tUsing feature(s)", temp_list, "accuracy is", percentage)
                
                # new higher accuracy is computed so we can save that and add current feature to list
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    feature_to_add = j
        
        # add new feature to add to feature list, making sure it isn't zero/empty
        if (feature_to_add != 0): 
            current_features.append(feature_to_add)
            best_accuracy_percentage = "{:.2%}".format(best_accuracy)
            # print new best feature set to user with its accuracy
            print("Feature set", current_features, "was best, accuracy is", best_accuracy_percentage)
        else:
            break
    
    # search is done so we can output the final accuracy and feature list to the user
    final_accuracy_percentage = "{:.2%}".format(best_accuracy)
    print("\nForward selection search done. The best feature subset is", current_features, "which has an accuracy of", final_accuracy_percentage)
    

# function for performing forward selection search on features and dataset 
def backwardSearch(data):
    # create feature list containing all possible features; we will be removing features one by one
    current_features = []
    for i in range(1, len(data[0])):
        current_features.append(i)
    best_accuracy = 0
    
    # for loop to traverse down search tree
    for i in range(len(data)):
        feature_to_remove = 0
        
        # for loop to traverse through each feature
        for j in range(1, len(data[i])):
            # consider removing new feature if it has not been considered already
            if (j in current_features):
                # compute accuracy of new feature set list using cross validation function
                # making sure to note we are removing a feature from the feature list
                accuracy = crossValidation(data, current_features, j, "remove")
                # for printing feature set to user only
                temp = current_features.copy()
                temp.remove(j)
                percentage = "{:.2%}".format(accuracy)
                # print accuracy to user when attempting to use current set of features
                print("\tUsing feature(s)", temp, "accuracy is", percentage)
                
                # new best accuracy is computed so we can save it and note to remove feature j
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    feature_to_remove = j
        
        # remove feature j from feature set list, making sure it is not empty/zero
        if (feature_to_remove != 0): 
            current_features.remove(feature_to_remove)
            best_accuracy_percentage = "{:.2%}".format(best_accuracy)
            # output new best feature set list and its accuracy
            print("Feature set", current_features, "was best, accuracy is", best_accuracy_percentage)        
        else:
            break
        
    # backward search is done so we can output the best feature set list for this dataset and its accuracy
    final_accuracy_percentage = "{:.2%}".format(best_accuracy)
    print("\nBackward selection search done. The best feature subset is", current_features, "which has an accuracy of", final_accuracy_percentage)
        

# program starts here 
# outputs a menu to the user, asking them which dataset file to use and which algorithm to use
print("Welcome to my Feature Selection Algorithm.")
file_name = str(input("Type in the name of the file to test: "))
print("Type the number of the algorithm you want to run.")
print("\t1) Forward Selection")
print("\t2) Backward Elimination")
algo = str(input())

# import data and store values in list for easier access
data_set = pd.read_csv(file_name, sep="  ", engine='python', header=None)
data = data_set.values.tolist()

# begin search on dataset; print to user number of features and instances in the dataset
print("\nThis dataset has", len(data[0])-1, "features, with", len(data), "instances.")
if (algo == "1"):
    print("Beginning forward selection search...")
    featureSearch(data)
elif (algo == "2"):
    print("Beginning backward elimination search...")
    backwardSearch(data)

