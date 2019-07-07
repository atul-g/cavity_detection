# Classifying images of teeth with and without cavity

This is a self-project on making a CNN on classifying whether a teeth has cavities or not using real time camera captured images instead of X-ray images.
I created my own dataset by downloading images from Google under different Search Keywords. I used the Google Image Downloader extension in Firefox browser. 

[Link](https://addons.mozilla.org/en-US/firefox/addon/google-images-downloader/) to the extension.
For [Chrome](https://chrome.google.com/webstore/detail/image-downloader/cnpniohnfphhjihaiiggeabnkjhpaldj).

The dataset I created is provided in the repository.
It contains in total 884 images of 2 classes. They have been split into train and test folders in 8:2 ratio.

I used Tensorflow hub's Imagenet v3 to create the initial layers and added a dense layer after that to create the neural network.
