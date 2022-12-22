import platform
import tensorflow
import time, pickle
import keras
print("Platform: {}".format(platform.platform()))
print("Tensorflow version: {}".format(tensorflow.__version__))
print("Keras version: {}".format(keras.__version__))


from keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython.display import Image
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
import keras.applications
import os

# from google.colab import drive
# drive.mount('/content/drive')


# ! unzip /content/drive/MyDrive/Dataset_OpenCvDl_Hw2_Q5.zip > /dev/null 2>&1
dog=os.listdir(r"/content/Dataset_OpenCvDl_Hw2_Q5/training_dataset/Cat")
cat=os.listdir(r"/content/Dataset_OpenCvDl_Hw2_Q5/training_dataset/Dog")
print("dog:", len(dog))
print("cat:", len(cat))
x = ["dog","cat"]        # 水平資料點
h = [len(dog),len(cat)]   # 高度
plt.title('Class Distribution')
plt.ylabel("Number of images")
def createLabels(data):
    for item in data:
        height = item.get_height()
        plt.text(
            item.get_x()+item.get_width()/2., 
            height*1, 
            '%d' % int(height),
            ha = "center",
            va = "bottom",
        )
A=plt.bar(x,h)
createLabels(A)

plt.show()

from keras.preprocessing.image import ImageDataGenerator
train_data = ImageDataGenerator(rescale=1. / 255)
validation_data = ImageDataGenerator(rescale=1. / 255)
test_data = ImageDataGenerator(rescale=1. / 255)

train = train_data.flow_from_directory(
    '/content/Dataset_OpenCvDl_Hw2_Q5/training_dataset',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary')

valid = validation_data.flow_from_directory(
    '/content/Dataset_OpenCvDl_Hw2_Q5/validation_dataset',
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary')

print('train steps:', len(train))
print('valid steps:', len(valid))


# !pip install tensorflow-addons
# import tensorflow_addons as tfa


nb_epoch=3


# 捨棄 ResNet50 頂層的 fully connected layers
net = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(224,224,3))
x = net.output
x = Flatten()(x)


# 增加 Dense layer
output_layer = Dense(1, activation='sigmoid', name='sigmoid')(x)
net_final = Model(inputs=net.input, outputs=output_layer)

# loss_function=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.4,gamma=1.0)
loss_function=keras.losses.BinaryCrossentropy()
optimizer=keras.optimizers.Adam(learning_rate=8e-5)
net_final.compile(optimizer=optimizer, loss=loss_function,metrics=['accuracy', 'mse'])

# net_final.summary()
hist=net_final.fit(
  train,
  validation_data=valid,
  epochs=nb_epoch
)
net_final.save('Dog_Cat_Binary_epoch_3.h5')
