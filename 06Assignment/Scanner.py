import csv
import numpy as np

def scan(file):

    # create the variables that will be used
    training_data = []
    training_dicts = []
    y_vals = []

    with open(file) as csvfile:
        # Read in the CSV
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        max_cols = 0
        col_min = 1

        # fill in variables with CSV data
        for row in reader:
            y_vals.append(float(row.pop(0)))
            t_dict = {}
            for col in row:
                if col == "":
                    continue
                splitStr=col.split(":")
                key = int(splitStr[0])
                val = float(splitStr[1])
                t_dict[key]=val
                col_min = min(col_min,val)
                max_cols = max(max_cols,key,len(row))
            training_dicts.append(t_dict)

        for t_dict in training_dicts:

            data_row = []

            for i in range(max_cols):
                inx = i + col_min
                if inx in t_dict.keys():
                    data_row.append(t_dict[inx])
                else:
                    data_row.append(0.0)
            training_data.append(data_row)

    return {'d': training_data, 'l': y_vals}