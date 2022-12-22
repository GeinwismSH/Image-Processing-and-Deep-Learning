from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# import torch
from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
import keras.applications
import random
# from torchsummary import summary
# from torchvision import models
# from torchvision import transforms
import keras
import tensorflow as tf
import keras
import matplotlib.image as img
from PIL import Image

from hw2_5ui import Ui_MainWindow



class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton_01.clicked.connect(self.load_image)
        self.ui.pushButton_2.clicked.connect(self.process5_1)
        self.ui.pushButton_3.clicked.connect(self.process5_2)
        self.ui.pushButton_4.clicked.connect(self.process5_3)
        self.ui.pushButton_5.clicked.connect(self.process5_4)
        self.ui.pushButton_6.clicked.connect(self.process5_5)
    #讀取圖片
    def load_image(self):
        global filepath1

        filename1, filetype1 = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")                 
        print(filename1, filetype1)
        
        filepath1=filename1

                
    #影像處理區
    def process5_1(self):
        image1 = cv2.imread('./inference_dataset/Cat/8043.jpg')
        image2 = cv2.imread('./inference_dataset/Dog/12053.jpg')
        
        img1=cv2.resize(image1, (224, 224))
        img2=cv2.resize(image2, (224, 224))
        # image1 = img.imread('./inference_dataset/Cat/8043.jpg') # 讀取圖片
        # image2 = img.imread('./inference_dataset/Dog/12053.jpg') # 讀取圖片
        # img1=image1.resize((224,224))
        # img2=image2.resize((224,224))

        plt.subplot(2,2,1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title("Cat")
        plt.subplot(2,2,2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title("Dog")
        plt.tight_layout()                                    
        plt.show()    


  
    def process5_2(self):
        img=cv2.imread("./hw2_5_2.png")
        cv2.imshow("2.5.2",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def process5_3(self):
        # 捨棄 ResNet50 頂層的 fully connected layers
        #避免額外下載weight，這裡把ImageNet改None
        net = keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None,
                    input_shape=(224,224,3))
        x = net.output
        x = Flatten()(x)
        # 增加 Dense layer，以 softmax 產生個類別的機率值
        output_layer = Dense(1, activation='sigmoid', name='sigmoid')(x)
        net_final = Model(inputs=net.input, outputs=output_layer)
        net_final.summary()

    def process5_4(self):
        img = cv2.imread('./Accuracy Comparison.png')
        cv2.imshow("2.5.5",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def process5_5(self):
        mymodel=keras.models.load_model("./Dog_Cat_Binary_epoch_3.h5")

        img = cv2.imread(filepath1)
        img_resize=cv2.resize(img, (224, 224))/255.0

        # p=[[float(j) for j in i] for i in mymodel(tf.constant([img_resize]))]
        p=mymodel(tf.constant([img_resize]))
        pp=float(p[0])

        print(pp)
        if pp>0.5:
            classname="Dog"
        else:
            classname="Cat"
        # a="Confidence = "+str(round(pp,2))+"\n"+"Prediction Label : "+str(classname)
        a="Prediction  : "+str(classname)
        plt.title(a)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        pass
        


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())