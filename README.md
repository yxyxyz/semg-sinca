# SINCA
A lightweight model architecture using **S**pectral **IN**ception Encoder and **C**hannel **A**ttention for sEMG-based hand gesture recognition.

The repository is the implementation of the EMBC 2026 paper: ``Lightweight Learnable Spectral Inception Network for Hand Gesture Recognition from SEMG Signals``.

## Install & Dependencies
We use ``conda`` to set the related environment and dependencies. The code was tested in Ubuntu 24.04.
```
conda create -n env python=3.12
pip install numpy, scipy, pandas, matplotlib, torch, torchvision 
```

## Structure
```
semg-sinca
│   README.md	
│   .gitgnore
│   comparison.py # reproduction of pca+svm and cnn-lstm
│   data_utils.py # segment data, feature extraction, dataset, etc.
│   train.py # training functions and classes served for running experiments
│   model.py # models, include sine, cacnn, sinca
│   main.py # experiments
│
└───data # download ninapro data and unzip there, named DB2/DB_s1, etc.
│      
└───runs # saved tensorboard, checkpoint, csv training data
│      
└───save # related to papers

```

* You can use ``tensorboard`` to check each subject after training. 
* save dir provides ``mean_std.py`` used for the report.

## Usage
To run the scripts, you can use the following command with specific arguments
```
python -m main 
```

## Training & Evaluation
Training and evaluation are in ``train.py``. Feel free to modify or use them.

## Citation 
We would greatly appreciate it if you cite our EMBC 2026 paper when using this repository
```
@inproceedings{yang2026sinca,
  title={Lightweight Learnable Spectral Inception Network for Hand Gesture Recognition from SEMG Signals},
  author={Yang, Yuxin and Ren, Shiqi and Wu, Minliang},
  booktitle={2026 48th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  year={2026},
  organization={IEEE}
}
```
