import pandas as pd
import random

# import dataset
data = pd.read_csv('Small_Data.txt', sep="  ", engine='python', header=None)
    
def testing():
    row = data.loc[0]

    # testing how to iterate through dataset
    # prints first row of dataset
    print(row)
    print()

    # prints first column of dataset
    for i in range(len(data)):
        print(data[0][i])
    

def leave_one_out_cross_validation(data, current_set, feature_to_add):
    return random.random()  # for testing purposes
    
def feature_search():
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
                accuracy = leave_one_out_cross_validation(data, current_features, j+1)
                
                if (accuracy > best_accuracy):
                    best_accuracy = accuracy
                    feature_to_add = j
                    
        current_features.append(feature_to_add)
        print("On level", i, "i added feature", feature_to_add, "to current set")
        

feature_search()