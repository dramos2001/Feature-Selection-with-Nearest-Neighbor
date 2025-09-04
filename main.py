import pandas as pd
import matplotlib.pyplot as plt
import math
import time


# function to plot data points from both feature sets chosen by algorithm
def scatterPlot(data, features):
    feature1 = features[0]
    feature2 = features[1]
    
    class_1 = [row for row in data if row[0] == 1.0]
    class_2 = [row for row in data if row[0] == 2.0]
    
    plt.scatter([row[feature1] for row in class_1], [row[feature2] for row in class_1], color='blue', label='Class 1')
    plt.scatter([row[feature1] for row in class_2], [row[feature2] for row in class_2], color='red', label='Class 2')
    
    plt.title(f'Scatter Plot of Feature {feature1} vs Feature {feature2}')
    plt.xlabel(f'Feature {feature1}')
    plt.ylabel(f'Feature {feature2}')
    plt.legend()
    plt.grid(True)
    plt.show()

# function for calculating the accuracy of our classifier
def crossValidation(data, current_features, feature, add_or_remove):
    feature_set = current_features.copy()
    # add current feature if doing forward search and remove if doing backward elimination
    if (add_or_remove == "add"):
        feature_set.append(feature)
    else:
        feature_set.remove(feature)
    
    # stores the number of correctly classified instances
    correctly_classified = 0
    
    # loop through entire dataset
    for i in range(len(data)):
        # make a list of objects in dataset to classify
        object_to_classify = data[i][1:]       
        # save the label/classification of the previous list of objects (1 or 2)
        label_object_to_classify = data[i][0]
        
        nearest_neighbor_distance = math.inf
        nearest_neighbor_location = -1
        
        # loop through entire dataset to determine euclidean distances and accuracy
        for k in range(len(data)):        
            # make sure not to compare the same feature to itself    
            if k != i:
                # calculate euclidean distance only for selected features in feature_set
                distance = 0
                
                for feature_index in feature_set:
                    # list object_to_classify adjusts for the fact that feature indices in feature_set correspond 
                    # to columns in the data (start from 1), so we subtract 1 to align it with 0-based indexing
                    distance += (object_to_classify[feature_index-1] - data[k][feature_index]) ** 2
                                
                # closer neighbor found to current feature so it must be saved
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
        
        if nearest_neighbor_location == -1:  # safeguard
            continue
        
        nearest_neighbor_label = data[nearest_neighbor_location][0]
        
        # nearest neighbor is of the same label/classification so we can increment num of correctly classified
        if (label_object_to_classify == nearest_neighbor_label):
            correctly_classified += 1
        
    # return accuracy to search algorithm
    return correctly_classified / len(data)

    
# function for performing forward selection search on features in dataset 
def featureSearch(data):
    start_time = time.time()
    
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
    # calculate time taken to compute algorithm; output to console
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds...")
    
    scatterPlot(data, current_features)
    

# function for performing forward selection search on features and dataset 
def backwardSearch(data):
    start_time = time.time()
    
    # create feature list containing all possible features; we will be removing features one by one
    current_features = []
    for i in range(1, len(data[0])):
        current_features.append(i)
        
    best_features = current_features.copy()
    global_best_accuracy = 0
    
    # loop to traverse down search tree and find best features
    while len(current_features) > 1:
        best_accuracy = 0
        feature_to_remove = 0
        best_subset = []
        
        for j in current_features:
            # compute accuracy of new feature-set list using cross validation function
            # make sure to note we are removing a feature from the feature list
            accuracy = crossValidation(data, current_features, j, "remove")
            # for printing feature subset to console
            temp_list = current_features.copy()
            temp_list.remove(j)
            percentage = "{:.2%}".format(accuracy)
            # print accuracy to user when attempting to use current feature subset
            print("\tUsing feature(s)", temp_list, "accuracy is", percentage)
            
            # if new best accuracy is computed, save it and remove feature j from current features list
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                feature_to_remove = j
                best_subset = temp_list
                
        # remove feature j from feature subset; make sure it is not empty/zero
        if feature_to_remove != 0:
            current_features.remove(feature_to_remove)
            best_percentage = "{:.2%}".format(best_accuracy)
            print("Feature set", current_features, "was best, accuracy is", best_percentage)
            
            # update global best subset and accuracy if this round improved
            if best_accuracy > global_best_accuracy:
                global_best_accuracy = best_accuracy
                best_features = best_subset
        else:
            break
        
    
    # backward search is done so we can output the best feature subset and its accuracy
    print("\nBackward elimination done. The best feature subset is", best_features, f"which has an accuracy of {global_best_accuracy:.2%}")
    # calculate time take to compute algorithm and output to console
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds...")
    
    scatterPlot(data, best_features)
        

def main():
    # outputs a menu to the user, asking them which dataset file to use and which algorithm to use
    print("Welcome to my Feature Selection Algorithm.")
    file_name = str(input("Type in the name of the file to test: "))
    # import data and store values in list for easier access
    # check that the inputted file exists
    try:
        data_set = pd.read_csv(file_name, sep="  ", engine='python', header=None)
        data = data_set.values.tolist()
        print(type(data))
    except FileNotFoundError:
        print("File not found. Please check the filename and try again.")
        return
    
    # user is now asked to select the algorithm
    print("Type the number of the algorithm you want to run.")
    print("\t1) Forward Selection")
    print("\t2) Backward Elimination")
    algo = str(input())
    # check that input is valid i.e. '1' or '2'
    if algo not in ['1', '2']:
        print("Invalid selection. Please choose 1 for Forward Selection or 2 for Backward Elimination")
        return

    # begin search on dataset; print to user number of features and instances in the dataset
    print("\nThis dataset has", len(data[0])-1, "features, with", len(data), "instances.")
    if (algo == "1"):
        print("Beginning forward selection search...")
        featureSearch(data)
    elif (algo == "2"):
        print("Beginning backward elimination search...")
        backwardSearch(data)

if __name__ == "__main__":
    main()
