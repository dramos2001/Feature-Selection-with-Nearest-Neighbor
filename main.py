import pandas as pd
import random

# import dataset
data = pd.read_csv('Small_Data.txt', sep="  ", engine='python', header=None)
    
# for testing purposes only
def testing():
    row = data.loc[0]

    # testing how to iterate through dataset
    # prints first row of dataset
    print(row)
    print()

    # prints first column of dataset
    for i in range(len(data)):
        print(data[0][i])
    

def crossValidationDemo():
    for i in range(len(data)):
        object_to_classify = data
        label_object_to_classify = data[i][0]
        
        print("Looping over i, at the", i, "location")
        print("The", i, "th object is in class", label_object_to_classify)

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
        
        
# featureSearch()
crossValidationDemo()