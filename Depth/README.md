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


## **Results**

Below are the results of applying different loss functions. First Row Corresponds to Target Depths. Second Row Corresponds to Predicted Depths

**i** --> BCEwithLogitLoss   (Combination of sigmoid and binary cross entropy loss)

**ii** --> RMSE    (Root Mean Square Loss)

**iii** ---> SSIM+BCEwithLogitLoss ( 1 x BCEwithLogitLoss + 2 xSSIM )

**iv** --> SSIM (Structural Similarity loss - compares local region of target pixel between reconstructed and original images)

<p align = "center"><img height = "500" src = "https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/depth.jpg"</p>
  
## **Inference**

1. Above are the results that ran for 10 EPOCHS with 80K dataset. 
2. From above we can see that BCEwithLogitLoss is giving good results than compared to other.
3. But BCEwithLogitLoss images are not sharp. It is more smooth, where as SSIM detects edges, gradients in a better way.
4. RMSE images are very blurry.
5. Thought the combination Of SSIM and BCEWithLogitLoss work well. But didnot get good results. May be, the weights used in not propere. Took 1:2 ratio. This can be fine tuned to get better results.
6. By all these results, for combined model chose BCEwithLogit Loss and SSIM Loss Function to play around.
6
  





