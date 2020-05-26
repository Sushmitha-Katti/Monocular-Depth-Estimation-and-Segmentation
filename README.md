# Monocular-Depth-Estimation-and-Segmentation

## For any AI/ML problem there will be some generic steps to perform.
1. Probelm statement.
2. Data Collection and preprocessing
3. Model Selection
4. Model Evaluation
5. If Model is not good repeat previous steps else stop.


**We will go step by step**


# **Problem Statement**

Without knowing a problem statement we cannot build the appropriate Model. SO what is our problem statement?

   **Build a DNN Model that takes background, background+foreground as a input and gives the depth and mask of them as a output** 
    
Interesting!But why do we need to build it? What is its use?

The goal of depth estimation is to obtain a representation of the spatial structure of a scene, recovering the three-dimensional shape and appearance of objects in imagery.

 Mask detection hepls in removal of background, its main use in medical field where we can use detect the tumereous cells, etc...
 
 So we now got the problem statement. Lets work on it.
 

# **Data Preparation**
  Data is like a fuel in AI/ML. Without it nothing happens. Took 1000 fg and 100bg images, overalapped fg on bg on 20 random positions, created masks and depths of them.
  
  For more info can refer this link. Detail Explaination is provided there.
  
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
   We just need to give how many we need to extract. It extracts all.
  
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
    
       *Couldn't try much transformation. The doubt I had was if we apply transformations to input then output will change. Output is dependent on input. Since it is not just object detection. However I could have applied transformations like brightness, Saturation which doesn't cause change in position of image*
       
       * **Bringing it to dataset formaat**[Refer here for code](https://github.com/Sushmitha-Katti/PyTNet/blob/master/Dataset/MaskDepth.py)
       
          1. Convert the whole data to raw data.
          2. Then split it according to the givene ratio. By default it is 70:30
          3. Input each subset for transformation. 
          4. Return
          Syntax
                          
              RawDataSet(train_split = 70,test_transforms = None,train_transforms = None, set_no=1, url_path ='None', whole_data = True )
             * whole_data - do all these for complete data. This is considered as priority. If it is false. Then look for url path and set_no
             * set_no - Do this for specific data, out of 5 sets.
          
  
 
 
 
  


## This is just a gist. Will document the whole process shortly.
