echo "Install R environment and packages..."
sudo apt install r-base-core
sudo R --vanilla -e 'install.packages("optparse", repos="http://cran.us.r-project.org")'
echo "This script uses the ClusterATAC tool to reproduce the performance comparison results in the article. In order to run efficiently, we do not retrain the neural network. If you need to download the original data and retrain the network, please run this command
: python ClusterATAC.py -m preprocess"
echo "Cancer subtyping using ClusterATAC, the number of categories is set to 18:"
python ClusterATAC.py -m cluster -n 18
echo "Cancer subtyping using ClusterATAC, the number of categories is set to 22:"
python ClusterATAC.py -m cluster -n 22
echo "Load other algorithm results and evaluate each method using the significance of the difference in survival:"
python ClusterATAC.py -m eval
