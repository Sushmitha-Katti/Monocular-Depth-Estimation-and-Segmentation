# **Mask Prediction**

## **Model**


<p align = "center"><img src = "https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/unet-mask.png"></p>

**There are three main parts in the model**
1. **DobleConv** - Conv->Batch Normalisation -> Relu -> Conv -> Batch Normalisation -> Relu
2. **Down** - MaxPool -> DoubleConv
3. **Up** - Upsample(bilinear) -> DoubleConv

**Parameters Count**
* Total Parameters : 4,321,473
* Trainable Parameters : 4,321,473
* Non Trainable Parameters : 0
* Input Size(MB) : 0.09
* Forward/Backward Pass(MB) : 21.47
* Params Size(MB) : 16.49
* Estimated total Size(MB) : 46.05

**Hyper Parameters** 
* Optimiser : Adam
* Scheduler : ReduceLrOn Pleateau
* Loss Functions : Dice Loss, BCEwithLogitLoss
* Total Epochs : 10
* LR : 0.01
* Evaluation Metrics : Dice Coefficent


## **Results**

Below are the results of applying different loss functions. First Row Corresponds to Target Masks. Second Row Corresponds to Predicted Masks

**i** --> BCEwithLogitLoss   (Combination of sigmoid and binary cross entropy loss)

**ii** --> Dice Loss    (1 - 2 x Intersection / Union)



<p align = "center"><img height = "500" src = "https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/mask.jpg"</p>
<p align = "center">To View the images properly, click on the image</p>
  
## **Inference**

1. Above are the results that ran for 10 EPOCHS with 80K dataset. 
2. From the above we can see that BCEwithLogitLoss mask color is gray, but it shows the whole image, where in Dice loss, the images are clear and whiter but Some of the regions are not detected. They are black.
3. Dice Coefficient of dice loss is higher than dice loss. But decided to go with BCEwithLogitLoss. 
4. The Dice Coefficient is less, because the pixel values were not matching, but visually BCEwithLogitLoss was more appealing. 
