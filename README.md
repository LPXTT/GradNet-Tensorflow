# GradNet-Tensorflow
This is the official implementation with training code for 'GradNet: Gradient-Guided Network for Visual Object Tracking' (ICCV2019 Oral). For more details, please refer to:

Introduction
--------------------------------
We propose a GradNet to update the template in single object tracking based on template information and gradients.</br>
[Results on OTB100](https://drive.google.com/file/d/1jvjZlQmDGqvrVgQuA4yZIsgZGZUws1_z/view?usp=sharing) </br>

![](https://github.com/LPXTT/GradNet-Pytorch/blob/master/GradNet.png)  

Requirements
--------------------------
1. Tensorflow
2. CUDA 8.0 and cuDNN 6.0
3. Python 2.7 / Python 3.6

Usage
--------------------------
### Train
1. Data preparation: Please refer to https://github.com/bertinetto/siamese-fc for details and change the data paths in parameters.py. 
2. Please run `$(ROOT_PATH)/train.py` to get your own model.
   
### Test
  Please run `$(ROOT_PATH)/track.py` for demo.
  
License
--------------------
Licensed under an MIT license.

Citation
--------------------
If you find GradNet useful in your research, please kindly cite our paper:</br>

    @InProceedings{GradNet_ICCV2019,
    author = {Peixia Li, Boyu Chen, Wanli Ouyang, Dong Wang, Xiaoyun Yang, Huchuan Lu},
    title = {GradNet: Gradient-Guided Network for Visual Object Tracking},
    booktitle = {ICCV},
    month = {October},
    year = {2019}
    }

Contact
--------------------
If you have any questions, please feel free to contact pxli@mail.dlut.edu.cn

Acknowledgments
------------------------------
Many parts of this code are adopted from other related works ([tensorflow-siamese-fc](https://github.com/www0wwwjs1/tensorflow-siamese-fc))
