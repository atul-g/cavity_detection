# Classifying images of teeth with and without cavity

This is a self-project on making a CNN to classify whether a teeth has cavities or not using real time (phone) camera captured images instead of X-ray images.
I created my own dataset by downloading images from Google under different Search Keywords. I used the Google Image Downloader extension in Firefox browser. 

[Link](https://addons.mozilla.org/en-US/firefox/addon/google-images-downloader/) to the extension (firefox).

For [Chrome](https://chrome.google.com/webstore/detail/image-downloader/cnpniohnfphhjihaiiggeabnkjhpaldj).

The dataset can be downloaded from [here](https://drive.google.com/file/d/1SWxNftwN6HDNNOP2ONiSd9kvlikm3jLS/view?usp=sharing).
The dataset contains in total 884 images of 2 classes. They have been split into train and test folders in 8:2 ratio.

The dataset structure:


|--Dataset

    |--train
  
      |--cavity (389 images)
      |--no_cavity (319 images)
      
    |--test
  
      |--cavity (97 images)
      |--no_cavity (79 images)
  

I used Tensorflow hub's Imagenet v3 to create the initial layers and added a dense layer after that to create the neural network.
It's important to make sure that you use the 2.0 version of Tensorflow as the pre-trained model that I have used (Imagenet v3) is for Tensorflow 2.0 version and won't work if Tensorlfow 1.x version is installed. I used this model as it was giving a much higher accuracy as compared to the 1.x models and the keras.application models.

The final accuracy values after training it in Colab for 15 epochs are:

`loss: 0.3825 - acc: 0.8771 - val_loss: 0.6718 - val_acc: 0.7955`

Although it yields a good accuracy value in both training and validation set the model still is not performing well enough when tasked with predicting new images. This is mainly because of the low quality of the dataset.
