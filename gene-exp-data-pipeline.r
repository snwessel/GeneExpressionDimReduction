# Download gene expression data from a legacy GDC database
# Align against the geneome reference hg19.
query <- GDCquery(project = "TCGA-GBM",
                  data.category = "Gene expression",
                  data.type = "Gene expression quantification",
                  platform = "Illumina HiSeq", 
                  file.type  = "normalized_results",
                  experimental.strategy = "RNA-Seq",
                  legacy = TRUE)

# Download files (about 32MB of data)
GDCdownload(query, 
            method = "api", 
            files.per.chunk = 10, 
            directory = "C:/Users/Sarah/Desktop/Repos/GeneExpressionDimReduction/data")



