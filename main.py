import pandas as pd
import random
import math

file = "Small_Data_96.txt"

# import dataset
data = pd.read_csv(file, sep="  ", engine='python', header=None)
    
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
    

def crossValidationDemo():
    correctly_classified = 0
    
    for i in range(len(data)):
        row_to_classify = data.loc[i]
        object_to_classify = []
        for j in range(1, len(row_to_classify)):
            object_to_classify.append(row_to_classify[j])
        label_object_to_classify = data[0][i]
        
        nearest_neighbor_distance = math.inf
        nearest_neighbor_location = math.inf
        
        for k in range(len(data)):
            print("Ask if", i, "is nearest neighbor with", k)
            
            if (k != i):
                distance = math.sqrt(object_to_classify - data)  # fix this
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
        
        if (label_object_to_classify == nearest_neighbor_label):
            correctly_classified += 1
            
        print("Object", i, "is class", label_object_to_classify)
        print("Its nearest neighbor is", nearest_neighbor_location, "which is in class", nearest_neighbor_label)                    
        # print("Looping over i, at the", i, "location")
        # print("The", i, "th object is in class", label_object_to_classify)

def leaveOneOutCrossValidation(data, current_set, feature_to_add):
    return random.random()  # for testing purposes
    
def featureSearch():
    current_features = []
    
    # for loop to traverse down search tree
    for i in range(len(data)):
        print("On the " + str(i) + "th level of the search tree")
        feature_to_add = 0
        best_accuracy = 0
        row = data.loc[i]
        
        # for loop to traverse through each feature
        for j in range(1, len(row)):
            # consider adding new feature if it has not been considered already
            if (j not in current_features):
                print("--Considering adding the", row[j], "feature")
                accuracy = leaveOneOutCrossValidation(data, current_features, j+1)
                
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    feature_to_add = j
                    
        current_features.append(feature_to_add)
        print("On level", i, "i added feature", feature_to_add, "to current set")
        
        
    print(current_features)
featureSearch()
# crossValidationDemo()