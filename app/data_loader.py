import csv
import os 
import numpy as np

class DataLoader:
  def __init__(self):
    pass

  def load_data_files(self):
    """Load the reads from each of the normalized_results files in the data directory.
      Return a dataframe containing the file contents"""
    X = np.empty((20532,0), int)
    # iterate through the files
    path = ".\\data\\TCGA-GBM\\legacy\\Gene_expression\\Gene_expression_quantification"
    for dirname in os.listdir(path):
      for filename in os.listdir(path + "\\" + dirname):
        # read from the tsv file
        tsv_file = open(path + "\\" + dirname + "\\" + filename)
        reader = csv.reader(tsv_file, delimiter="\t")
        gene_exp_values = []
        for row in reader:
          gene_exp_values.append(row[1])
        # append to our matrix
        X = np.append(X, np.array([gene_exp_values]).T, axis=1)

    print("Created a matrix with shape: ", X)
    return X

  