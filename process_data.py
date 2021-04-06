import csv
import gzip
import numpy as np
import time
import os

def process_data_files():
  """Load the reads from each of the normalized_results files in the data directory.
    Return a dataframe containing the file contents"""
  # initialize variables
  X = np.empty((0, 60482), int) # 20531?
  start_time = time.perf_counter()
  # for each project
  data_path = ".\\data"
  sample_ids = []
  for project_name in ["TCGA-LAML", "TCGA-HNSC"]: #os.listdir(data_path):
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
  np.savetxt("processed-data/metadata.csv", np.array(sample_ids), delimiter=",")
  
  # write to CSV
  np.savetxt("processed-data/gene-expression.csv", X, delimiter=",")

  # log things
  seconds_passed = int(time.perf_counter() - start_time)
  print("Finished after", seconds_passed, "seconds.")
  print("Output a dataframe with shape", )


process_data_files()