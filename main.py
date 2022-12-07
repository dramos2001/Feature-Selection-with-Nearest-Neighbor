import pandas as pd
import random
import math
from copy import deepcopy

file = "Small_Data_96.txt"

# import dataset
data_set = pd.read_csv(file, sep="  ", engine='python', header=None)
data = data_set.values.tolist()

# for testing purposes only
def testing():
    row = data.loc[0]

    # prints len of dataset i.e. number of rows
    print(len(data))

    # testing how to iterate through dataset
    # prints first row of dataset
    print(row)
    print()

    # prints first column of dataset
    for i in range(len(data)):
        print(data[0][i])
    

def crossValidationDemo(data, current_features, feature):
    feature_set = current_features.copy()
    if (feature > 0):
        feature_set.append(feature)
    else:
        feature_set.remove(feature)
    
    data_copy = deepcopy(data)
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            if (j not in feature_set):
                data_copy[i][j] = 0
    
    correctly_classified = 0
    
    for i in range(len(data)):
        object_to_classify = data[i][1:]
        label_object_to_classify = data[i][0]
        
        nearest_neighbor_distance = math.inf
        nearest_neighbor_location = math.inf
        
        for k in range(1, len(data)):
            #print("Ask if", i, "is nearest neighbor with", k)
            
            if (k != i):
                distance = math.sqrt(sum([(a-b) ** 2 for a, b in zip(object_to_classify, data_copy[k][1:])])) # i think this should be good
                
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data_copy[nearest_neighbor_location][0]
        
        if (label_object_to_classify == nearest_neighbor_label):
            correctly_classified += 1
            
        #print("Object", i, "is class", label_object_to_classify)
        #print("Its nearest neighbor is", nearest_neighbor_location, "which is in class", nearest_neighbor_label)                    
        # print("Looping over i, at the", i, "location")
        # print("The", i, "th object is in class", label_object_to_classify)
        
    accuracy = correctly_classified / len(data_copy)
    return accuracy

    
# function for performing forward selection search on features and dataset 
def featureSearch(data):
    current_features = []
    best_accuracy = 0
    
    # for loop to traverse down search tree
    for i in range(len(data)):
        # print("On the " + str(i) + "th level of the search tree")
        feature_to_add = 0
        
        # for loop to traverse through each feature
        for j in range(1, len(data[i])):
            # consider adding new feature if it has not been considered already
            if (j not in current_features):
                accuracy = crossValidationDemo(data, current_features, j)
                # print accuracy to user when attempting to use current set of features
                print("Using feature(s)", current_features, "accuracy is", accuracy)
                
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    feature_to_add = j
        
        if (feature_to_add != 0): 
            current_features.append(feature_to_add)
            print("On level", i, "i added feature", feature_to_add, "to current set")
            if (len(current_features) == len(data[0])):
                break
        else:
            print("Accuracy decreasing, stop search here")
            break
    
    # fix formatting of percent later    
    print("Forward selection search done. The best feature subset is", current_features, "which has an accuracy of", accuracy)
        

def menu():
    print("Welcome to my Feature Selection Algorithm.")
    file_name = str(input("Type in the name of the file to test: "))
    print("Type the number of the algorithm you want to run.")
    print("     1) Forward Selection")
    print("     2) Backward Elimination")
    algo = str(input())
    
    # import data and convert to list
    data_set = pd.read_csv(file_name, sep="  ", engine='python', header=None)
    data = data_set.values.tolist()
    
    if (algo == "1"):
        featureSearch()
    elif (algo == "2"):
        # backwardElimination()
        print()
    else:
        print("Error. Invalid input")
    
    
featureSearch(data)
