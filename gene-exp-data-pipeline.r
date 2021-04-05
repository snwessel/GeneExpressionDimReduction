# Download gene expression data from a legacy GDC database
# Documentation for GDCQuery here: https://rdrr.io/bioc/TCGAbiolinks/man/GDCquery.html 
query <- GDCquery(project = "TCGA-LAML", # Required, iterate through projects to get more
                  data.category = "Transcriptome Profiling",
                  data.type = "Gene Expression Quantification",
                  workflow.type = "HTSeq - FPKM",
                  #sample.type = "Primary Blood Derived Cancer - Bone Marrow", 
                  experimental.strategy = "RNA-Seq")

# Download files (about 546MB of data)
GDCdownload(query, 
            method = "api", 
            files.per.chunk = 20, 
            directory = "C:/Users/Sarah/Desktop/Repos/GeneExpressionDimReduction/data")

data <- GDCprepare(query,
                   directory = "C:/Users/Sarah/Desktop/Repos/GeneExpressionDimReduction/data")
metadata <- as.data.frame(colData(data))

# process columns containing lists
cleaned_data <- subset(metadata, select=-c(treatments, primary_site, disease_type))

# write the metadata to CSV
filename = "C:/Users/Sarah/Desktop/Repos/GeneExpressionDimReduction/metadata.csv"
write_csv(
  cleaned_data,
  filename,
  na = "NA",
  append = FALSE,
  col_names = TRUE,
  quote_escape = "double",
  eol = "\n"
)

