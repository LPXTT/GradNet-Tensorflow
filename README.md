# GradNet-Pytorch
This is the official implementation with training code for GradNet (ICCV2019 Oral). For more details, please refer to:


__GradNet: Gradient-Guided Network for Visual Object Tracking__

The code is coming soon...

GradNet: Gradient-Guided Network for Visual Object Tracking
=========================================
This is the official implementation with training code for GradNet (ICCV2019 Oral). For more details, please refer to:

Introduction
--------------------------------
We propose a GradNet to update the template in single object tracking based on template information and gradients.</br>
[Results on OTB100](https://drive.google.com/file/d/1jvjZlQmDGqvrVgQuA4yZIsgZGZUws1_z/view?usp=sharing) </br>

![](https://github.com/bychen515/ACT/blob/master/ACT.png)  

Requirements
--------------------------
1. Tensorflow 1.4.0 (Train) and Pytorch 0.3.0 (Test)
2. CUDA 8.0 and cuDNN 6.0
3. Python 2.7

Usage
--------------------------
### Train
  1. Please download the `ILSVRC VID dataset`, and put the `VID` folder into `$(ACT_root)/train/` </br>
  (We adopt the same videos as [meta_trackers](https://github.com/silverbottlep/meta_trackers). You can find more details in `ilsvrc_train.json`.)
  2. Run the `$(ACT_root)/train/DDPG_train.py` to train the 'Actor and Critic' network.
### Test
  Please run `$(ACT_root)/tracking/run_tracker.py` for demo.
  
License
--------------------
Licensed under an MIT license.

Citation
--------------------
If you find ACT useful in your research, please kindly cite our paper:</br>

    @InProceedings{Chen_2018_ECCV,
    author = {Chen, Boyu and Wang, Dong and Li, Peixia and Wang, Shuang and Lu, Huchuan},
    title = {Real-time 'Actor-Critic' Tracking},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    month = {September},
    year = {2018}
    }

Contact
--------------------
If you have any questions, please feel free to contact bychen@mail.dlut.edu.cn

Acknowledgments
------------------------------
Many parts of this code are adopted from other related works ([py-MDNet](https://github.com/HyeonseobNam/py-MDNet) and [meta_trackers](https://github.com/silverbottlep/meta_trackers))
