# iterate through the projects we want to query
projects <- list("TCGA-UVM", "TCGA-PCPG", "TCGA-KIRC", "TCGA-HNSC", "TCGA-LAML")
for (project_name in projects) {
  
  # Download gene expression data from a legacy GDC database
  # Documentation for GDCQuery here: https://rdrr.io/bioc/TCGAbiolinks/man/GDCquery.html 
  query <- GDCquery(project = project_name, # Required, iterate through projects to get more
                    data.category = "Transcriptome Profiling",
                    data.type = "Gene Expression Quantification",
                    workflow.type = "HTSeq - FPKM",
                    #sample.type = "Primary Blood Derived Cancer - Bone Marrow", 
                    experimental.strategy = "RNA-Seq")
  
  # Download files (about 546MB of data)
  data_path = paste("C:/Users/Sarah/Desktop/Repos/GeneExpressionDimReduction/data/raw-downloads/", 
                    project_name, sep="")
  GDCdownload(query, 
              method = "api", 
              files.per.chunk = 20, 
              directory = data_path)
  
  data <- GDCprepare(query,
                     directory = data_path)
  metadata <- as.data.frame(colData(data))

  # process columns containing lists
  cleaned_data <- subset(metadata, select=-c(treatments, primary_site, disease_type))

  # write the metadata to CSV
  filename = paste("C:/Users/Sarah/Desktop/Repos/GeneExpressionDimReduction/data/metadata/", project_name, ".csv", sep="")
  write_csv(
    cleaned_data,
    filename,
    na = "NA",
    append = FALSE,
    col_names = TRUE,
    quote_escape = "double",
    eol = "\n"
  )
  
  # write the gene expression values to CSV
  assay_data = as.data.frame(assay(data))
  filename = paste("C:/Users/Sarah/Desktop/Repos/GeneExpressionDimReduction/data/gene-expression/", project_name, ".csv", sep="")
  write_csv(
    assay_data,
    filename,
    na = "NA",
    append = FALSE,
    col_names = TRUE,
    quote_escape = "double",
    eol = "\n"
  )
}
