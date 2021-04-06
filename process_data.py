import csv
import gzip
import numpy as np
import time
import os

project_names = ["TCGA-LAML", "TCGA-HNSC"]
processed_metadata_filename = "processed-data/metadata.csv"

def process_data_files():
  """Load the reads from each of the normalized_results files in the data directory.
    Return a dataframe containing the file contents"""
  # initialize variables
  X = np.empty((0, 60482), int) # 20531?
  start_time = time.perf_counter()
  # for each project
  data_path = ".\\data"
  sample_ids = []
  for project_name in project_names:
    print("Processing data for", project_name, "...")
    # data\TCGA-LAML\TCGA-LAML\harmonized\Transcriptome_Profiling\Gene_Expression_Quantification
    project_data_path = data_path + "\\" + project_name + "\\" + project_name + "\\harmonized\\Transcriptome_Profiling\\Gene_Expression_Quantification"
    # iterate through the files
    for dirname in os.listdir(project_data_path):
      for filename in os.listdir(project_data_path + "\\" + dirname):
        # read from the tsv file
        with gzip.open(project_data_path + "\\" + dirname + "\\" + filename, mode="rt") as tsv_file:
          reader = csv.reader(tsv_file, delimiter="\t")
          next(reader)
          gene_exp_values = []
          for row in reader:
            gene_exp_values.append(float(row[1]))
          # append as a row to our matrix
          X = np.append(X, np.array([gene_exp_values]), axis=0)
          sample_ids.append(dirname)
  
  # write the list of sample ids (in order) to CSV
  np.savetxt(processed_metadata_filename, np.array(sample_ids), delimiter=",", fmt='%s')
  
  # write to CSV
  np.savetxt("processed-data/gene-expression.csv", X, delimiter=",")

  # log things
  seconds_passed = int(time.perf_counter() - start_time)
  print("Finished after", seconds_passed, "seconds.")
  print("Output a dataframe with", X.shape)

def process_metadata():
  # load the saved sample ids
  sample_ids = np.genfromtxt(processed_metadata_filename, delimiter=",", dtype=str)
  print("Loaded sample ids with shape", sample_ids.shape)

  # put all of the classes in order based on the sample ids
  ordered_diagnoses = np.full(sample_ids.shape[0], "")
  #combined = np.append(sample_ids, np.full(sample_ids.shape[0], ""))
  #print("combined shape:", combined.shape)
  for project_name in project_names:
    # load the existing metadata files
    filename = ".\\metadata\\" + project_name + ".csv"
    print("Reading from", filename)
    with open(filename) as csvfile:
      reader = csv.reader(csvfile, delimiter=",")
      diag_index = next(reader).index("primary_diagnosis")
      for row in reader:
        diagnosis = row[diag_index].split(", ")[0]
        sample_id = row[7]
        #print(sample_id, diagnosis)
        ordered_diagnoses = np.where(sample_ids==sample_id, diagnosis, ordered_diagnoses)
        print("Found at ids", np.nonzero(sample_ids==sample_id))
  print(sample_ids[:5])
  print(ordered_diagnoses)

#process_data_files()
process_metadata()