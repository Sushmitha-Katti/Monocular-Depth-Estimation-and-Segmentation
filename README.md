# Monocular-Depth-Estimation-and-Segmentation

## DataSet

Used Custom prepared dataset to train model

Link for Custom Data - [Click here](https://github.com/Sushmitha-Katti/EVA-4/tree/master/Session14)

### Sample Images

## Data Statistics

* fg-bg -- 400k
* bg --- 100
* mask ---400k
* depth -- 400k

## Input
* fg-bg, bg
## Output
* mask
* Depth

## Model
Modified UNET
Split at the head

## Agumentations
* Resize
* Normalise
* To tensor
## Loss Functions
* dice Loss
* SSIM
* BCEwithlogits
* RMSprop
## Evaluation Metrics
* Dice Coefficent
## Optimiser
* Adam
## Scheduler
* ReduceLrOnPleateau


## This is just a gist. Will document the whole process shortly.
