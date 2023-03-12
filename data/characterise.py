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
def characterise():
    csvs = get_csvs()
    for file in csvs:
        df = pd.read_csv(file)
        print(f"{file}: \nrows: {df.shape[0]} \ncols: {df.shape[1]}")

characterise()