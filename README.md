# pytorch-retinanet-pipeline

This repo is an implementation of RetinaNet using pytorch adapted from the following [code](https://github.com/yhenon/pytorch-retinanet). 

## Installation

pip install mmcv
pip install tqdm
pip install torchvision
pip install scikit-image

##  Data pre-processing
The code handles the use of the UAV123 dataset. The dataset can be downloaded at the following [link](https://cemse.kaust.edu.sa/ivul/uav123).

Select the directory where you want to save the dataset. Once the dataset is downloaded, we need to convert the format to use the retinanet.
Run the uav2retina script as follow :
    
    python data-preprocessing/uav2retina.py -i /path/to/uav123/folder -o /path/to/save/new/annotations

##  Training

##  Evaluation
