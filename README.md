# ClusterATAC
ClusterATAC is a cancer subtype tool based on ATAC-seq profiles. The input for the framework high-dimensional ATAC-peak scores of all the tumor samples. The output is the corresponding subclass label for each sample. ClusterATAC is mainly divided into two components: 1. GAN-based feature extraction module is used to obtain abstract features from deep learning using high-dimensional original input. 2. A GMM-based clustering module for determining the number of types and the class labels corresponding to each sample. 3. Based on the original data, apply the comparison algorithms to achieve the typing results.  
```{r}
# the input raw data file is all.txt and runs the following command to finish all processes: 
python ClusterATAC.py -i ./all.txt  
# the Clustering output file are stored in ./results/all.clusteratac  
```
Specifically, for the feature extraction module:
```{r}
python ClusterATAC.py -m feature -i ./all.txt  
# the low-dimensional features encoded by the neural network are stored in ./fea/all.clusteratac  
```
ClusterATAC's GMM clustering module is used as follows:  
```{r}
python ClusterATAC.py -m cluster -n 22 -i ./all.txt  
# record the corresponding class label for each sample and the output file is ./results/all.clusteratac 
```
ClusterATAC's performance comparison module (using the autoencoder method as an example) is used as follows: 
```{r} 
python ClusterATAC.py -m ae -i ./all.txt
# record the corresponding class label for each sample and the output file is ./ TCGA_ATAC_peak_Log2Counts_dedup_sample.spectral
```  
ClusterATAC is based on the Python program language. The generative adversarial network's implementation was based on the open-source library scikit-learn 0.21.3, Keras 2.2.4, and Tensorflow 1.14.0 (GPU version). After testing, this framework has been working correctly on Ubuntu Linux release 18.04. Due to the high dimensionality of the raw data, the size of the neural network is enormous. We used the NVIDIA TITAN XP (12G) for the model training. When the GPU's memory is not enough to support the running of the tool, we suggest simplifying the encoder's network structure.
