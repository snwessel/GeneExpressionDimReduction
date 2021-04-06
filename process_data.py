import csv
import gzip
import numpy as np
import os
import pandas as pd
import time

project_names = ["TCGA-LAML", "TCGA-HNSC"]


def combine_gene_expression_files():
  df = pd.DataFrame(columns=[])
  for project_name in project_names:
    print("Processing data for", project_name, "...")
    # load data from the project file
    project_filename = ".\\data\\gene-expression\\" + project_name + ".csv"
    project_df = pd.read_csv(project_filename)
    # save to our combined dataframe
    print("\tproject shape:", project_df.shape)
    df = pd.concat([df, project_df], axis=1)
    print("\tcombined shape:", df.shape)
  # save the dataframe to csv
  np.savetxt("data/processed-data/gene-expression.csv", df.to_numpy().T, delimiter=",")
  # save the ordered barcodes to csv
  barcodes = np.array(df.columns).T
  np.savetxt("data/processed-data/ordered-barcodes.csv", barcodes, delimiter=",", fmt='%s')


def process_metadata():
  # load the saved sample ids
  sample_ids = np.genfromtxt("data/processed-data/ordered-barcodes.csv", delimiter=",", dtype=str)
  print("Loaded sample ids with shape", sample_ids.shape)

  # put all of the classes in order based on the sample ids
  ordered_diagnoses = np.full(sample_ids.shape[0], "")
  for project_name in project_names:
    # load the existing metadata files
    filename = "data/metadata/" + project_name + ".csv"
    print("Reading from", filename)
    with open(filename) as csvfile:
      reader = csv.reader(csvfile, delimiter=",")
      diag_index = next(reader).index("primary_diagnosis")
      for row in reader:
        diagnosis = row[diag_index].split(", ")[0]
        sample_id = row[0]
        ordered_diagnoses = np.where(sample_ids==sample_id, diagnosis, ordered_diagnoses)
  # log the number of each diagnosis
  (unique, counts) = np.unique(ordered_diagnoses, return_counts=True)
  print("Diagnosis counts:", np.asarray((unique, counts)).T)
  # save to file
  np.savetxt("data/processed-data/ordered-diagnoses.csv", ordered_diagnoses.T, delimiter=",", fmt='%s')


combine_gene_expression_files()
#process_metadata()