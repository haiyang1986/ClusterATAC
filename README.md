#ClusterATAC
ClusterATAC is a cancer subtype tool based on ATAC-seq profiles. The input for the framework high-dimensional ATAC-peak scores of all the tumor samples. The output is the corresponding subclass label for each sample. ClusterATAC is mainly divided into two components: 1. GAN-based feature extraction module, which is used to obtain abstract features from deep learning using high-dimensional original input. 2. A GMM-based clustering module for determining the number of types and the class labels corresponding to each sample. 3. Based on the original data, apply the comparison algorithms to achieve the typing results.  
Specifically, for the feature extraction module, the input raw data file is TCGA_ATAC_peak_Log2Counts_dedup_sample.txt and runs the following command:  
python ClusterATAC.py -m feature -i ./TCGA_ATAC_peak_Log2Counts_dedup_sample.txt  
The low-dimensional features encoded by the neural network are stored in ./fea/TCGA_ATAC_peak_Log2Counts_dedup_sample.fea  
ClusterATAC's GMM clustering module is used as follows:  
python ClusterATAC.py -m cluster -n 22 -i ./TCGA_ATAC_peak_Log2Counts_dedup_sample.txt  
Output the corresponding class label for each sample. The output file is ./ TCGA_ATAC_peak_Log2Counts_dedup_sample.out  
ClusterATAC's performance comparison module (using the spectral clustering method as an example) is used as follows:  
python ClusterATAC.py -m compare -p spectral -i ./TCGA_ATAC_peak_Log2Counts_dedup_sample.txt  
Output the corresponding class label for each sample. The output file is ./ TCGA_ATAC_peak_Log2Counts_dedup_sample.spectral  
ClusterATAC is based on the Python pregame language. The implementation of the generative adversarial network was based on the open source library Keras 2.2.4 and Tensorflow 1.12.0 (GPU version). After testing, this framework has been working properly on CentOS Linux release 7.6. Due to the high dimensionality of the raw data, the size of the neural network is huge. We used the NVIDIA TITAN XP (12G) for the model training. When the GPU's memory is not enough to support the running of the tool, we suggest simplifying the network structure of the encoder.
