# Adaptive Synthetic Sampling Approach for Imbalanced Learning 

ADASYN is a python module that implements an adaptive oversampling technique for skewed datasets.

## Developed under [IMEC](http://www.imec-nl.nl/nl_en/netherlands.html) 

### Dependencies
* pip (needed for install)
* numpy
* scipy
* scikit-learn


### Installation

To use ADASYN you will need to clone this repository and install it running the following :

    git clone https://github.com/stavskal/ADASYN.git
    cd ADASYN
    pip install -r requirements.txt
    
    
After you have installed the packages you can proceed with using:

    from adasyn import ADASYN
    adsn = ADASYN(k=7,imb_threshold=0.6, ratio=0.75)
    new_X, new_y = adsn.fit_transform(X,y)  # your imbalanced dataset is in X,y

    
    
    
Original paper can be found [here](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4633969&url=http://ieeexplore.ieee.org/xpls/abs_all.jsp%3Farnumber%3D4633969) 

This module implements the idea presented in the paper by Haibo He et al. and also includes oversampling for **multiclass** classification problems. It is designed to be compatible with [scikit-learn] (https://github.com/scikit-learn/scikit-learn). It focuses on oversampling the examples that are harder to classify and has shown results which sometimes outperform SMOTE or SMOTEboost.

An example can be seen below:

![alt tag](https://github.com/stavskal/ADASYN/blob/master/sample.jpg)


Props to [fmfn](https://github.com/fmfn) who implemented different oversampling techniques for his good code structure, which highly influenced this module, and documentation
