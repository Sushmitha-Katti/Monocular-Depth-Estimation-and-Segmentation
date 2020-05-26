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

    **Build a DNN Model that takes background, background+foreground as a input and gives the depth and mask of them as a output.** 
    
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
  
  Since data is spread across 5 zip folders. Needed a stratergy to extract all of them.

"""
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
    print(f"data set extraction took {round(end-start,2)}s") """
  
  


## This is just a gist. Will document the whole process shortly.
