# **Depth Prediction**

Model


<p align="center"><img src = "https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/unet-depth.png"></p>

**There are three main parts in the model**
1. DobleConv - Conv->Batch Normalisation -> Relu -> Conv -> Batch Normalisation -> Relu
2. Down - MaxPool -> DoubleConv
3. Up - Upsample(bilinear) -> DoubleConv

**Parameters Count**
* Total Parameters : 4,367,937
* Trainable Parameters : 4,367,937
* Non Trainable Parameters : 0
* Input Size(MB) : 0.09
* Forward/Backward Pass(MB) : 23.1
* Params Size(MB) : 16.66
* Estimated total Size(MB) : 40.67

**Hyper Parameters** 
* Optimiser : Adam
* Scheduler : ReduceLrOn Pleateau
* Loss Functions : SSIM, BCEwithLogitLoss, SSIM+BCE, RMS
* Total Epochs : 10
* LR : 0.01

**Results**

Below are the results of applying different loss functions.

**i** --> BCEwithLogitLoss   

**ii** --> RMSE    

**iii** ---> SSIM+BCE    

**iv** --> SSIM
<p align = "center"><img height = "500" src = "https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/depth.jpg"</p>
  





