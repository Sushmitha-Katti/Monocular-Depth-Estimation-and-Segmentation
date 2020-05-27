# **Code Explanation**

All the code is implemented [here](https://github.com/Sushmitha-Katti/PyTNet). **PyTNet, A deep neural net Libraby Built on top of Pytorch**

    extract_data(no_of_set = 1): <a href="https://github.com/Sushmitha-Katti/PyTNet" title="Code">lodash</a>

Unzips the data from drive. It takes input as no_of_set. Since the data is spread across 5 zips. It asks how many sets need to be extracted. To play with data you can extract one or 2 sets. To extract whole data we should give as 5. Above that it gives error
 
 
    AlbumentationTransforms( transforms):
 
Applies transformation. This uses Albumentation library

    RawDataSet(train_split = 70,test_transforms = None,train_transforms = None, set_no=1, url_path ='None', whole_data = True)

Converts the raw data to data set format. It takes the whole and convert into dataset format. Again it splits into train_split: total_len-train_split ratio. Then apply transformations and return.

Here we can state for how many data we need to do. That is given by whole_data. If this is true. All the 5 sets are included. Else it checks set_no and base url(url_path) and converts only that set.


    load(trainset,testset,seed=1,batch_size=128,num_workers=4,pin_memory=True):

loads the data using data loader according to batch size.

    show_sample_data(trainloader)

Displays sample input and target data

    train_model(model,device,trainloader,testloader,optimizer,mask_criterion, depth_criterion,EPOCHS,scheduler = False,batch_scheduler = False ,best_loss = 1000,path = "/content/gdrive/My Drive/API/bestmodel.pt"):

 Used to train model. Inturn it calls train and test function, trains the model for specified number of epochs and save the model if its loss is less than the best loss
 
    show_results(model,testloader,name):

Displays the results and save them. Here name refers to title of image.

    dice_coefficient(pred, target, mask=False):
    
 Gives the dice coefficient

    plot_curve(curves,title,Figsize = (7,7)):

Plots the given curves


