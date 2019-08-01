#!/usr/bin/env bash
#echo "Install R environment and packages..."
#R --vanilla -e 'install.packages("optparse", repos="http://cran.us.r-project.org")'
#R --vanilla -e 'install.packages("survival", repos="http://cran.us.r-project.org")'
echo "This script uses the ClusterATAC tool to reproduce the performance comparison results in our article. To run efficiently, we do not retrain the neural network. If you need to download the original data and retrain the neural network, please run this command: python ClusterATAC.py -m preprocess"
echo "Cancer subtyping using ClusterATAC, the number of categories is set to 18:"
python ClusterATAC.py -m cluster -n 18
echo "Cancer subtyping using ClusterATAC, the number of categories is set to 22:"
python ClusterATAC.py -m cluster -n 22
echo "Load other algorithm results and evaluate each method using the significance of the difference in survival:"
python ClusterATAC.py -m eval
echo "Finally, we get the result file of the cancer typing and the survival analysis result files, which are TCGA_ATAC_peak_Log2Counts_dedup_sample.clusteratac18, TCGA_ATAC_peak_Log2Counts_dedup_sample.clusteratac22, and survival.pval respectively."