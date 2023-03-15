# python file to characterise the dataset

# import libraries
import pandas as pd
import os

#get csvs of cwd
def get_csvs():
    cwd = os.getcwd()
    csvs = [] # list of the csvs
    for file in os.listdir(cwd):
        if file.endswith(".csv"):
            csvs.append(file)
    return csvs

# print(get_csvs())

# for each csv file, read it in and output some statistics 
# include:
    # count of each class (show imbalanced dataset) - show models overfit to majority class in imbalanced dataset
    # count of rows and columns (wavelengths)
    # range of numbers 
    # number of 0s 
    # number of nulls
    # any negative numbers?
def characterise():
    csvs = get_csvs()
    for file in csvs:
        df = pd.read_csv(file)
        print(f"{file}: \nrows: {df.shape[0]} \ncols: {df.shape[1]}")

characterise()