import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
import time

project_names = ["TCGA-LAML", "TCGA-HNSC", "TCGA-KIRC", "TCGA-UVM", "TCGA-PCPG"]


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

def generate_train_test_data():
  """Further process the 'processed' data to generate a matrix X and vector y to use in training and testing.
    Results should be saved to `X.csv` and `y.csv`."""
  # load the processed data using numpy from ordered-diagnoses.csv and gene-expression.csv
  print("Loading processed data from file...")
  labels = np.genfromtxt("data/processed-data/ordered-diagnoses.csv", delimiter=",", dtype=str)
  gene_exp = np.genfromtxt("data/processed-data/gene-expression.csv", delimiter=",")
  # filter out data with low-frequency labels
  print("Filtering low-frequency labels")
  (unique_labels, counts) = np.unique(labels, return_counts=True)
  for i in range(len(unique_labels)):
    # if the label count is less than 20, remove all instances of it 
    if counts[i] < 40:
      original_size = labels.shape[0]
      label_filter = labels != unique_labels[i]
      labels = labels[label_filter]
      gene_exp = gene_exp[label_filter]
      print("\tRemoved", original_size - labels.shape[0], "records with label", unique_labels[i])
  # log the new number of each diagnosis
  (unique, counts) = np.unique(labels, return_counts=True)
  print("New diagnosis counts:", np.asarray((unique, counts)).T)

  # embed labels using sklearn's LabelEncoder
  le = preprocessing.LabelEncoder()
  le.fit(labels)
  encoded_labels = le.transform(labels)

  # undersample the data (so all label counts are the same)
  undersampled_X, undersampled_y = RandomUnderSampler().fit_resample(gene_exp, encoded_labels)

  # save cleaned data to data/train-test-data folders
  np.savetxt("data/train-test-data/y.csv", undersampled_y, delimiter=",", fmt="%u ")
  np.savetxt("data/train-test-data/X.csv", undersampled_X, delimiter=",")

start_time = time.perf_counter()
#combine_gene_expression_files()
#process_metadata()
generate_train_test_data()
print("Operation took", int(time.perf_counter() - start_time), "seconds.")