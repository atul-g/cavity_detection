import os
import numpy as np
from zipfile import ZipFile


#extracting the zip file
zipobj=ZipFile(os.path.abspath('.')+'/dental_dataset.zip', 'r')
zipobj.extractall()

#Need to make sure you are using a 2.O version of Tensorflow
#!pip install tf-nightly-gpu
#!pip install "tensorflow_hub==0.4.0"


################################## IMAGE AUGMENTATION ##################################
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen_train = ImageDataGenerator(rescale=1./255) #Can add the image augmentation parameters if needed

train_gen = image_gen_train.flow_from_directory(batch_size=20, directory = os.path.abspath('.')+'/train', shuffle = True, target_size =(224, 224), class_mode = 'binary')

image_gen_test = ImageDataGenerator(rescale = 1./255)

test_gen = image_gen_test.flow_from_directory(batch_size=20, directory=os.path.abspath('.')+'/test', target_size=(224, 224), class_mode='binary')


################################## MODEL CREATION ##################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras import layers
import tensorflow_hub as hub

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/3"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(224, 224,3))

feature_extractor.trainable = False #freezing the upper layer

model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(2, activation='softmax')
])

model.summary()

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
keras_layer (KerasLayer)     (None, 1280)              2257984   
_________________________________________________________________
dense (Dense)                (None, 2)                 2562      
=================================================================
Total params: 2,260,546
Trainable params: 2,562
Non-trainable params: 2,257,984
_________________________________________________________________
"""

model.compile(
  optimizer='adam', 
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
  

################################## EXECUTING THE MODEL ##################################
EPOCHS = 15
history = model.fit_generator(train_gen,
                    epochs=EPOCHS,
                    validation_data=test_gen)

################################## VISUALIZING THE MODEL PERFORMANCE ##################################

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

