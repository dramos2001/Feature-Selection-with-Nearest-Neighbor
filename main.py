import pandas as pd

# import dataset
data = pd.read_csv('Small_Data.txt', sep="  ", engine='python', header=None)
row = data.loc[0]

# testing how to iterate through dataset
# prints first row of dataset
print(row)
print()

# prints first column of dataset
for i in range(len(data)):
    print(data[0][i])