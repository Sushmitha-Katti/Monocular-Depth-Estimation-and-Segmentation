# **Mask And Depth Prediction**


## **Model**

<p align="center"><img src = "https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/combined.png"></p>

**There are three main parts in the model**
1. **DobleConv** - Conv->Batch Normalisation -> Relu -> Conv -> Batch Normalisation -> Relu
2. **Down** - MaxPool -> DoubleConv
3. **Up** - Upsample(bilinear) -> DoubleConv

**Parameters Count**
* Total Parameters : 8,67,8274
* Trainable Parameters : 8,67,8274
* Non Trainable Parameters : 0
* Input Size(MB) : 0.09
* Forward/Backward Pass(MB) : 46.38
* Params Size(MB) : 38.10
* Estimated total Size(MB) : 79.58

**Hyper Parameters** 
* Optimiser : Adam
* Scheduler : ReduceLrOnPleateau
* Loss Functions : SSIM, BCEwithLogitLoss, SSIM+BCE
* Total Epochs : 10(max)
* LR : 0.01
* Evaluation Metrics : Dice Coefficient.
* Total Count of Data trained : 400k

## **Implementation**

1. Trained on whole data(400k) using BCEwithLogitLoss for Mask and SSIM for Depth 


## **Results**

Below are the results of applying different loss functions. 
**i** --> BCEwithLogitLoss for mask. BCEwithLogitLoss for depth(trained for 7 epochs)

**ii** -->  BCEwithLogitLoss for mask. BCEwithLogitLoss for depth(trained for 10 epochs of each of the 5 sets)

**iii** ---> BCEwithLogitLoss for mask. SSIM for depth(trained for 7 epochs)

<p align = "center">Mask Results</p>

<p align = "center"><img height = "500" src = "https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/combined-mask.jpg"</p>
  
 
  
<p align = "center">Depth Results</p>
<p align = "center"><img height = "500" src = "https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/combined-deptj.jpg"</p>
  <p align = "center">To View the images properly click on the image</p>


For **i** and **ii**, I used similar loss function. But what is the difference. 

1. **i** is trained for 7 epochs using complete dataset once.
2. **ii** is trained using 80k dataset each time. Like transfer learning. 
   * My dataset is spread over 5 sets. In all the sets the images were similar. 
   * So taking the idea of transfer learning. Trained each epoch for 10 epochs at once. Saved the best model.
   * Then loaded the best model, and started training for next set.
   < Questions Posed are
   When resuming training for next set what should be the learning rate? Should it continue from where it stoped or start with the initial learning rate that was chosen for previous set.
   So to answer this question I tried both ways. For some sets I have contined where the LR stopped. For some I took as it was for initialised for previous one.
   If we start from where lr stopped previous, we may decay the LR. This is the the new set, the inital LR may be the better idea.


  
## **Inference**

1. From the above result **iii** method i.e BCEwithLogitLoss for mask and SSIM for depth is ruled out.
2. BCEwithLogit Loss will be good for both.
3. There is not much difference in **i** and **ii**. We can go with anyone. 

## **Links for Code Implementation**
1. [BCEwithLogitLoss](https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Depth/Depth_BCE.ipynb)
2. [RMSE](https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Depth/Depth_RMSE.ipynb)
3. [SSIM+BCE](https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Depth/Depth_SSIM%2BBCE.ipynb)
4. [SSIM](https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Depth/Depth_SSIM.ipynb)



