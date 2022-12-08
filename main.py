import pandas as pd
import math
from copy import deepcopy


def crossValidationDemo(data, current_features, feature, add_or_remove):
    feature_set = current_features.copy()
    if (add_or_remove == "add"):
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
        object_to_classify = data_copy[i][1:]        
        label_object_to_classify = data_copy[i][0]
        
        nearest_neighbor_distance = math.inf
        nearest_neighbor_location = math.inf
        
        for k in range(len(data)):            
            if (k != i):
                distance = sum([(a-b) ** 2 for a, b in zip(object_to_classify, data_copy[k][1:])])
                distance = math.sqrt(distance)
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data_copy[nearest_neighbor_location][0]
        
        if (label_object_to_classify == nearest_neighbor_label):
            correctly_classified += 1
        
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
                accuracy = crossValidationDemo(data, current_features, j, "add")
                # for printing purposes only
                temp_list = current_features.copy()
                temp_list.append(j)
                percentage = "{:.2%}".format(accuracy)
                # print accuracy to user when attempting to use current set of features
                print("\tUsing feature(s)", temp_list, "accuracy is", percentage)
                
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    feature_to_add = j
        
        if (feature_to_add != 0): 
            current_features.append(feature_to_add)
            best_accuracy_percentage = "{:.2%}".format(best_accuracy)
            print("Feature set", current_features, "was best, accuracy is", best_accuracy_percentage)
        else:
            break
    
    final_accuracy_percentage = "{:.2%}".format(best_accuracy)
    print("\nForward selection search done. The best feature subset is", current_features, "which has an accuracy of", final_accuracy_percentage)
    

# function for performing forward selection search on features and dataset 
def backwardSearch(data):
    current_features = []
    for i in range(1, len(data[0])):
        current_features.append(i)
    best_accuracy = 0
    
    # for loop to traverse down search tree
    for i in range(len(data)):
        # print("On the " + str(i) + "th level of the search tree")
        feature_to_remove = 0
        
        # for loop to traverse through each feature
        for j in range(1, len(data[i])):
            # consider removing new feature if it has not been considered already
            if (j in current_features):
                accuracy = crossValidationDemo(data, current_features, j, "remove")
                # for printing to user only
                temp = current_features.copy()
                temp.remove(j)
                percentage = "{:.2%}".format(accuracy)
                # print accuracy to user when attempting to use current set of features
                print("\tUsing feature(s)", temp, "accuracy is", percentage)
                
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    feature_to_remove = j
        
        if (feature_to_remove != 0): 
            current_features.remove(feature_to_remove)
            best_accuracy_percentage = "{:.2%}".format(best_accuracy)
            print("Feature set", current_features, "was best, accuracy is", best_accuracy_percentage)        
        else:
            break
        
    final_accuracy_percentage = "{:.2%}".format(best_accuracy)
    print("\nBackward selection search done. The best feature subset is", current_features, "which has an accuracy of", final_accuracy_percentage)
        

def menu():
    # print to user intro of program and ask for what algorithm they want to compute
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
    else:
        print("Error. Invalid input")

menu()
