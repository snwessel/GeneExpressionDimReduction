import csv
import os 
import pandas as pd

class DataLoader:
  def __init__(self):
    pass

  def load_data_files(self):
    """Load the reads from each of the normalized_results files in the data directory.
      Return a dataframe containing the file contents"""
    df = pd.DataFrame(columns=[])
    # iterate through the files
    path = ".\\data\\TCGA-GBM\\legacy\\Gene_expression\\Gene_expression_quantification"
    for dirname in os.listdir(path):
      for filename in os.listdir(path + "\\" + dirname):
        tsv_file = open(path + "\\" + dirname + "\\" + filename)
        reader = csv.reader(tsv_file, delimiter="\t")
        gene_exp_values = []
        for row in reader:
          gene_exp_values.append(row[1])
        df[filename] = gene_exp_values

    print("# files:", len(df.columns))
    return df

  