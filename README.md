# IJCAI-2019 Alibaba Adversarial AI Challenge

## Background

<img src="https://github.com/Tristan-YF/IJCAI-2019_Alibaba_Adversarial_AI_Challenge/blob/master/adversarial%20image.png" style="zoom:0.3"/>

Task: Targeted Adversarial Attack

The goal of the targeted attack is to slightly modify source image in a way that image will be classified as specified target class by generally unknown machine learning classifier.

final rank: 11/2519

[Competition home page](https://tianchi.aliyun.com/competition/entrance/231701/introduction?spm=5176.12281915.0.0.71a94b15lWaznZ)

## Method

Our method can be seen a  momentum iterative  gradient method(MIGM). In order to further improve the transferability on different models, we adopted a ensemble learning strategy. In addition, we proposed a self-adaptive strategy to  adaptively adjust the ensemble weight and disturbation scale according to our effect. To  resist some general defend method like denoise, we utilize the Gassian Kernel to smooth the disturbation to improve the robustness of our  method.



## Dependencies

Python 3.6

TensorFlow 1.4.1



## Pre-trained Model 

You can download three official pre-trained models (inception_v1, resnet_v1_50, vgg_16) from [here](https://tianchi.aliyun.com/competition/entrance/231701/information) and save them to checkpoints/
