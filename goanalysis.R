source("https://bioconductor.org/biocLite.R")
#installation of packages needed
biocLite(c('SummarizedExperiment', 'edgeR', 'geneplotter','org.Hs.eg.db','GOstats', 'xtable', 'statmod', 'Category'))
biocLite('biomaRt')
biocLite('AnnotationDbi')
biocLite('corpcor')
biocLite('grex')

install.packages('RSQLite')
install.packages('DBI')

#Import libraries
library(org.Hs.eg.db)
library(GOstats)
library(xtable)
library(statmod)
library(biomaRt)
library(corpcor)
library(grex)
library(DBI)
library(Category)


####PERC 95
#Load your gene of interest dset
genes <- read.csv('path of list of significant genes', header = TRUE, sep = ',', quote = '\"')
tissues <- colnames(genes) #tissues to iterate
totalgenes <- read.csv('path_list of genes used for analysis', header=FALSE) #gene universe loading


geneUni <- grex(totalgenes$V1) #get entrezID for gene universe


for (i in tissues){
  geneid <- grex(as.vector(genes[,i])) #get genes for a tissue with entrezID
  geneid <- geneid[, c('ensembl_id', 'entrez_id')] #get only entrezID for gene set
  geneUni <- geneUni[, c('ensembl_id', 'entrez_id')] # same for gene universe
  geneid <- geneid[complete.cases(geneid), ] #eliminate ones without GO terms or entrezID
  geneUni <- geneUni[complete.cases(geneUni), ] #eliminate ones without GO terms or entrezID

  params <- new("GOHyperGParams", geneIds=as.vector(geneid$entrez_id), universeGeneIds=as.vector(geneUni$entrez_id),
              annotation="org.Hs.eg.db", ontology="BP",
              pvalueCutoff=0.01, testDirection="over") #set up parameters, cut-off, ontology, etc etc.

  conditional(params) <- TRUE #condition the test to avoid parent terms highly enriched.

  hgOver <- hyperGTest(params) #run test

  goresults <- summary(hgOver) # get the result
  goresults <- goresults[goresults$Size >= 5 & goresults$Count >= 5, ] #filter to avoid terms with -inf odds ratio, this are noise
  goresults <- goresults[order(goresults$Pvalue, decreasing = FALSE), ] #order by pvalue
  head(goresults, n =10) #print top 10
  #GET the symbols and put them after the terms to see which genes of your list are inside
  geneIDs <- geneIdsByCategory(hgOver)[goresults$GOBPID] 
  geneSYMs <- sapply(geneIDs, function(id) select(org.Hs.eg.db, columns = "SYMBOL", key = id, 
                                                  keytype = "ENTREZID")$SYMBOL)
  geneSYMs <- sapply(geneSYMs, paste, collapse = ", ")
  goresults <- cbind(goresults, Genes = geneSYMs)
  rownames(goresults) <- 1:nrow(goresults)
  #print it in a table in html formated
  xtab <- xtable(goresults, display = c('d','s','g','f','f','d','d','s','s'), align = "l|c|r|r|r|r|r|p{3cm}|p{3cm}|")
  print(xtab, file = paste(c("path to store",i,"termination of file .html"), collapse =''), type = "html")
}




