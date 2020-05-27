 <p align="center"><img src ="https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/title.png"></p>

## For any AI/ML problem there will be some generic steps to perform.
1. Problem statement.
2. Data Collection and preprocessing
3. Model Selection
4. Model Evaluation
5. If Model is not good repeat previous steps else stop.


**We will go step by step**


# :small_orange_diamond: **Problem Statement**

Without knowing a problem statement we cannot build the appropriate Model. SO what is our problem statement?

|  "Build a DNN Model that takes background, background+foreground as a input and gives the depth and mask of them as a output" |
|-------------------------------------------------------------------------------------------------------------------------------|
<p align="center"><img width=500 src="https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/problem.jpg"></p> 

  
    
Interesting! But why do we need to build it? What is its use?

> The goal of depth estimation is to obtain a representation of the spatial structure of a scene, recovering the three-dimensional shape and appearance of objects in imagery.

> Mask detection hepls in removal of background, its main use in medical field where we can use detect the tumereous cells, etc...
 
 So we now got the problem statement. Lets work on it.
 
 # :small_orange_diamond: **Data Collection and Preprocessing**
  Data is like a fuel in AI/ML. Without it nothing happens. Took 100 fg and 100bg images, overalapped fg on bg on 20 random positions, created masks and depths of them.
  
  For more info can [refer this link](https://github.com/Sushmitha-Katti/EVA-4/tree/master/Session14){:target="_blank"}. Detail Explaination is provided there.
  
  Finally we had
  * Foreground Images - 100
  * Background Images - 100
  * Fg+Bg -400k
  * fg-Bg-mask - 400k
  * fg-bg-depth - 400k
  In a 5 zip folder of 80 images of 3 kinds each
  
  Now our data is ready. But we can't send it directly to the model. It should be  in the form that model takes it.
  
  1. **Since data is spread across 5 zip folders. Needed a stratergy to extract all of them.**


          def extract_data(no_of_set = 1):
            if(no_of_set > 6 or no_of_set < 1):
              print('No of sets should be not be less that 1 and greater that 6')
              return
            for i in range(1,no_of_set+1):
              start = time.time()
              if (os.path.isdir("data_"+str(i))):
                  print (f'Imagesof set {i} already downloaded...')
                  
              else:
                    archive = zipfile.ZipFile(f'/content/gdrive/My Drive/Mask_Rcnn/Dataset/data_part{str(i)}.zip')
                    archive.extractall()
                  end = time.time()
                print(f"data set extraction took {round(end-start,2)}s") 
   We just need to give how many sets we need to extract. It only extract that much.That number should be between 1 to 5
  
   **Advantages of zipping in 5 sets**
   
   * Zip file may be corrupted if we zip whole data in single file
   * If we want to play around with the data we can extract 1 or 2 sets instead of extracting whole data.
 
 2. **Data transformation and data loading**
       * The data need to be in the form which model accepts.
       * So it should undergo transformation.
       * Used **Albumentation** for transformation.
         * Resize - Image size was 224 x 224. But if we load the image like this only then it requires lot of memory and processing will also be slow. So Resized to 64 x 64.
         * Normalisation - To bring all images to same scale
         * To Tensor - Convert all images to tensor.
         
         
    
      > *Couldn't try much transformation. The doubt I had was if we apply transformations to input then output will change. Output is dependent on input. Since it is not just object detection. However I could have applied transformations like brightness, Saturation which doesn't cause change in position of image*
       
       * **Bringing it to dataset formaat** [Refer here for code](https://github.com/Sushmitha-Katti/PyTNet/blob/master/Dataset/MaskDepth.py)
       
          1. Convert the whole data to raw data which gives 4 outputs path of bg, fg-bg, mask depth. Bg was something different to deal with. Since it was not in zip and had only 100 images. It is spread accros each set. Simple trick to get it was to use ciel function
          
             *Each bg was spred around 800 fg-bg's. So we can use ciel(fg-bg_img_no/800). This gave a perfect result.*
          2. Then split it according to the givene ratio. By default it is 70:30
          3. Input each subset for transformation. Here both fg-bg and bg is concatenated to 1 image of 6 Channels 
          4. Return
          
          Syntax
                          
              RawDataSet(train_split = 70,test_transforms = None,train_transforms = None, set_no=1, url_path ='None', whole_data = True )
             * whole_data - do all these for complete data. This is considered as priority. If it is false. Then look for url path and set_no
             * set_no - Do this for specific data, out of 5 sets.
             
      * After this passed the returned values(traindata, testdata) to dataloader. This is the usual code as we use for any data loaders.
      [click here for the code](https://github.com/Sushmitha-Katti/PyTNet/blob/master/train_test_loader.py)
      
      ![Sample Input](https://github.com/Sushmitha-Katti/Monocular-Depth-Estimation-and-Segmentation/blob/master/Assets/input.jpg)
      
      
      Now the data is ready to go into the model. But which model? Will see in next section.
      
# :small_orange_diamond: **Model Selection**
    
   This is the interesting part. It poses lot of questions.
   
   * Which model to chose?
   * How to combine both models?
   * Till now we had labels as 0,1,2..... But now mask and depth.
   * How the model should be? How to convert object recognised model to mask and depth models?
     
   Till now we mostly delt with object recognisation. But now masks and segments. Thought that there may be complicated models designed for these problems. Started reading papers about mask and depth. Now broke the problem into 2 parts.
   
   1. Mask
   2. Depth.
   
   First we need to write a model to do seperate tasks, then we can think of combinig.
   
   1. **Mask** - 
   
      **Attempt 1**
       * Started with the **resent18** architecture. 
       * Removed avg pooling and fc layer. 
       * Since we should have input and output same size. The simple option I found was to make padding = 1
       * That was the disaster. Cuda out of Memory!
       * Since padding = 1 Every layer will have 64 x 64, which makes processing slower and takes lot of memory. Then I realised not only parameter count, memory storage, forward/ backward pass memory also matters.
       
       **Attempt 2**
       
       * In many papers, arcticles I read, lot were focusing on U-Net(Sounds something different). Started studing more of it. This model performed better for segmentation. This follows encoder-decoder architecture. So may be memory/storage and all will be less.It is also inspired from resnet. Will have many receptive fields. 
       
      * This is the architecture of U-Net
      
      * Used Adam as optimiser, reduce lr on pleateau as scheduler. Trained for 1 set of dataset(80k). The results were good. SO thought of continuing with this only.
      
   2. **Depth**
    
       * Since results for mask was good, tried U-Net architecture thought of trying same network for depth also
       * Only thing I changed was target in data preparation
       * After seeing the results I was amazed. It was predicting depth also(not a good results though)
       
     ***After that I realised that this Neural Networks are really crazy. They learn whatever we make them learn***
    
   3. **Combined model**
   
       **Question Posed**
       1. How to combine the model? 
       2. If we combine also how to make them learn? NN's will learn form backpropogation. Which loss can we backpropogate? Which weights to consider. Both targets should have different weights to learn
       
       Started searching for the answeres. Finally I learnt that 
       * we can split the model at somewhere in middle. By that it will learn both targets will be same. After that it will have different learning paths to optimise for specifically for that target.
       * Since every pass will be acumilated by gradients, by using backward, those gradients will only be backwarded. So we can use weighted loss function for backward
       
       
                 '''torch.backward.autograd([loss_1, loss_2])'''
                      
# :small_orange_diamond: **Model Evaluation**
   * This was the hardest part I have delt with.
  
  **Questions Posed**
  1. Which loss functions to use for mask and depth?
  2. Which Optimiser, Scheduler to use?
  3. How to Evaluate the accuracy? Since we need to compare the whole data. 
  
  * There were plenty of loss functions in the papers mentioned. Confused which one to use.
  * Decided to try with **Binary Cross Entropy with logits** and **Dice Loss** for mask.
  * **SSIM**,**Binary Cross Entropy with logits**, **RMSloss** for depth.
  * I learnt that for segmentations, depth kind of problems 'ADAM' , 'RMSPROP' would be better. I decided to stick to ADAM.
  * For Accuarcy there is no single method is bethod. 
    * Came across Pixel wise Comparision, IOU, Dice Coefficient methods.
    * The disadvantage of Pixel wise Comparision is, it compared pixel wise. Most of the time pixel values will not be same, it doesn't give the better accuracy.
    * Remaining IOU and dice Co-efficient. 
      * IOU - area of intersection/area of union
      * Dice- Coefficient - 2*area of intersection/ area of union(more like a f1 score)
      * Decided to go with dice coefficient. Since both are almost same.
      
 # :small_orange_diamond: **Implementation**
   1. **Mask** 
       * Model - Unet
       * Parameters - 4,321,473
       * Optimiser - Adam
       * Scheduler - Reduce Lr On Pleateau
       * Loss - Dice loss , BCE
       * Total No Of Epochs trained for - 10
       * Total No of data used - 80k(1 set)
  
   2. **Depth**
      * Model - Unet
      * Parameters - 4,367,937
      * Optimiser - Adam
      * Scheduler - Reduce Lr
      * Loss - BCE, SSIM, RMSProp, BCE+SSIM
      * Total No of Epochs trained for-10
      * Total number of data used - 80k(1 set)
    
      
   3. **Combined model**
   
       * Parameters - 8,678,274
       * Optimiser - Adam
       * Scheduler - Reduce Lr, Step Lr
       * Loss - BCE, SSIM
       * Total No of Epochs trained for - 10(max)
       * Total number of data used - 400K(whole data)
     
         
 # :small_orange_diamond: **Code Explaination**
 
 # :small_orange_diamond: **Future Work**
 * There is lot of improvisation that can be done here. Because of time constraint couldn't try much.
 * Tried only for adam, should try for other optimisers. Once tried for **SGD with one cycle LR** but the results were like ultrascans. So left it
 * Can Try Various loss functions, and their combinations also. Like weighted combination of diceloss and BCEwithlogitloss etc...
 * The evaluation metrics(Dice Coefficient) used is not giving appropriate results. They are in the range of 0.00's which is not at all good. Should find better evalation metric or should modify that only to give good results.
 * Tried only with ReduceLr and Step Lr Schedulers. It may work well with super convergence with correct hyperparameters.
 
 # :small_orange_diamond: **Key Learnings**
 * Patience is must to deal with AI/ML problems. With lot of disconnections, internet problems, should not be frustrated.
 * There are  2 fields while dealing the problem statement. Both are very important.
    1. Theoritical knowledge - Like which model to use, what happens when we use this etc. Breaking up the problem statement. Achiving good results in small problems then combine all those to achive the goal.
    2. Techincal Knowledge -Coding the theory parts, bringing them to reality. Simplifying the code.
 * Documentation. This is very important part. This is for us and others to understand our code.
 
 # :small_orange_diamond: **References**

 
      
      
  
   
       
       

       
       
   
     

      
          
  
 
 
 
  



