# SINCA
A lightweight model using **S**hort Time Fourier Transform **I**mitation **N**etwork and **C**hannel **A**ttentions for sEMG based hand gesture recognition.

# Install & Dependencies
We use ``conda`` to set the related environment and dependencies. The code was tested in Ubuntu 24.04.
```
conda create -n env python=3.12
pip install numpy, scipy, pandas, matplotlib, torch, torchvision 
```

# Structure
```
semg-sinca
│   README.md	
│   .gitgnore
│   data_utils.py # segment data, feature extraction, dataset, etc.
│   train.py # training functions and classes served for running experiments
│   model.py # models, include stft_imit_net, cacnn, sinca
│   main.py # exp1 & exp 3
│   finetune.py # exp 2
│
└───data # download ninapro data and unzip there, named DB2/DB_s1, etc.
│      
└───runs # saved tensorboard, checkpoint, csv training data
│      
└───save # related to papers

```

* You can use ``tensorboard`` to check each subject after training. 
* save dir provides ``mean_std.py`` used for the report.

# Usage
To run the scripts, you can use the following command with specific arguments
``
python -m main 
``
or
``
python -m finetune
``
## Algorithms & Models
We provided a pretrained model ``pretrained_sin_48_1-25.pth``. It is the STFT Imitation Network trained on subjects 1-25 repetitions 1, 3, 4 with 60 epochs 20000 batch size.

## Training & Evaluation
Training and evaluation are in ``train.py``. Feel free to modify or use them.
