# **Code Explanation**

All the code is implemented [here](https://github.com/Sushmitha-Katti/PyTNet). **PyTNet, A deep neural net Libraby Built on top of Pytorch**

<pre>
   extract_data(no_of_set = 1): <a href="https://github.com/Sushmitha-Katti/PyTNet/blob/master/evaluation_metrics/loss.py" title="Extract">Code</a>
</pre>

Unzips the data from drive. It takes input as no_of_set. Since the data is spread across 5 zips. It asks how many sets need to be extracted. To play with data you can extract one or 2 sets. To extract whole data we should give as 5. Above that it gives error
 
<pre>
    AlbumentationTransforms( transforms): <a href="https://github.com/Sushmitha-Katti/PyTNet/blob/master/Albumentationtransform.py" title="Extract">Code</a>
</pre>
 
Applies transformation. This uses Albumentation library

<pre>
    RawDataSet(train_split = 70,test_transforms = None,train_transforms = None, set_no=1, url_path ='None', whole_data = True) <a href="https://github.com/Sushmitha-Katti/PyTNet/blob/master/Dataset/MaskDepth.py" title="Extract">Code</a>
</pre>

Converts the raw data to data set format. It takes the whole and convert into dataset format. Again it splits into train_split: total_len-train_split ratio. Then apply transformations and return.

Here we can state for how many data we need to do. That is given by whole_data. If this is true. All the 5 sets are included. Else it checks set_no and base url(url_path) and converts only that set.

<pre>
    UNet(n_channels = 6,n_classes = 1): <a href = "https://github.com/Sushmitha-Katti/PyTNet/blob/master/Models/MaskDepthModel.py">Code</a>
</pre>

Mask and Depth Estimation Model

<pre>
    load(trainset,testset,seed=1,batch_size=128,num_workers=4,pin_memory=True): <a href="https://github.com/Sushmitha-Katti/PyTNet/blob/master/train_test_loader.py" title="Extract">Code</a>
</pre>

loads the data using data loader according to batch size.

<pre>
    show_sample_data(trainloader): <a href="https://github.com/Sushmitha-Katti/PyTNet/blob/master/show_images.py" title="Extract">Code</a>
 </pre>

Displays sample input and target data

<pre>
    train_model(model,device,trainloader,testloader,optimizer,mask_criterion, depth_criterion,EPOCHS,scheduler = False,batch_scheduler = False ,best_loss = 1000,path = "/content/gdrive/My Drive/API/bestmodel.pt"): <a href="https://github.com/Sushmitha-Katti/PyTNet/blob/master/Training/train_test_MnD.py" title="Extract">Code</a>
</pre>

 Used to train model. Inturn it calls train and test function, trains the model for specified number of epochs and save the model if its loss is less than the best loss
 
 <pre>
    show_results(model,testloader,name): <a href="https://github.com/Sushmitha-Katti/PyTNet/blob/master/Results/showMnD.py" title="Extract">Code</a>
  </pre>

Displays the results and save them. Here name refers to title of image.


<pre>
    dice_coefficient(pred, target, mask=False): <a href="https://github.com/Sushmitha-Katti/PyTNet/blob/master/evaluation_metrics/accuracy.py" title="Extract">Code</a>
 </pre>
    
 Gives the dice coefficient
 
<pre>
    plot_curve(curves,title,Figsize = (7,7)): <a href="https://github.com/Sushmitha-Katti/PyTNet/blob/master/evaluate.py" title="Extract">Code</a>
</pre>

Plots the given curves


