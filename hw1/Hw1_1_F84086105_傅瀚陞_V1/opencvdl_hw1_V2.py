from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np

from UI_main import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton.clicked.connect(self.open_file1) 
        self.ui.pushButton_2.clicked.connect(self.open_file2)
        self.ui.pushButton_3.clicked.connect(self.process1_1)
        self.ui.pushButton_4.clicked.connect(self.process1_2)
        self.ui.pushButton_5.clicked.connect(self.process1_3)
        self.ui.pushButton_9.clicked.connect(self.process1_4)
        self.ui.pushButton_8.clicked.connect(self.process2_1)
        self.ui.pushButton_7.clicked.connect(self.process2_2)
        self.ui.pushButton_6.clicked.connect(self.process2_3)
        

    #影像處理區
    def process1_1(self):
        self.img_path = filepath1 #得到路徑
        self.img = cv2.imread(self.img_path) #用opencv讀影像
        img=self.img.copy() #避免傷到原圖copy一份
        (B,G,R)=cv2.split(img) #分離出BGR三色
        zeros = np.zeros(img.shape[:2], dtype = np.uint8) #為了重新合成，需要和圖片大小相符的空0矩陣，.shape可得長寬厚，[:2]取前2
        img_R=cv2.merge([zeros,zeros,R])
        img_G=cv2.merge([zeros,G,zeros])
        img_B=cv2.merge([B, zeros, zeros])
        cv2.imshow("R",img_R)
        cv2.imshow("G",img_G)
        cv2.imshow("B",img_B)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
          
    def process1_2(self):
        self.img_path = filepath1
        self.img = cv2.imread(self.img_path)
        img=self.img.copy()

        #轉成灰階，by OpenCV Function
        img_gray_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("OpenCV Function",img_gray_1)

        #轉成灰階，by Average weighted
        (B,G,R)=cv2.split(img)
        
        #注意原圖格式為uint8，加總途中會超過256，因此先轉uint32再算
        B=np.array(B,dtype=np.uint32)
        G=np.array(G,dtype=np.uint32)
        R=np.array(R,dtype=np.uint32)
        I=np.array((B+G+R)/3,dtype=np.uint8)

        cv2.imshow("Average weighted",I)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process1_3(self):
        self.img_path = filepath1
        self.img = cv2.imread(self.img_path)
        img=self.img.copy()

        #轉hsv
        img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        #做綠色mask
        lower_green = np.array([45,150,50])
        upper_green = np.array([65,255,255])
        green_mask=cv2.inRange(img_hsv , lower_green , upper_green)

        #做白色mask
        lower_white = np.array([0,0,221])
        upper_white = np.array([180,30,255])
        white_mask=cv2.inRange(img_hsv , lower_white , upper_white)

        #做bitwise_and(img,img,dst,mask)
        bitwise_green=cv2.bitwise_and(img,img,dst=None,mask=green_mask)
        bitwise_white=cv2.bitwise_and(img,img,dst=None,mask=white_mask)

        cv2.imshow("bit_green",bitwise_green)
        cv2.imshow("bitwise_white",bitwise_white)

    def process1_4(self):
        self.img_path1 = filepath1
        self.img1 = cv2.imread(self.img_path1)
        self.img_path2 = filepath2
        self.img2 = cv2.imread(self.img_path2)
        img1=self.img1.copy()
        img2=self.img2.copy()

        output = cv2.addWeighted(img1, 1, img2, 0, 0)
        cv2.imshow("output",output)

        def blend(val):
            b=val/255
            a=1-b
            output = cv2.addWeighted(img1, a, img2, b, 0)
            cv2.imshow("output",output)
            
        cv2.createTrackbar('Blend', 'output', 0, 255,blend)
        cv2.setTrackbarPos('Blend', 'output', 0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process2_1(self):
        self.img_path = filepath1
        self.img = cv2.imread(self.img_path)
        img=self.img.copy()
        cv2.imshow("gaussian",img)

        def adjust(val):
            if val == 0:
                cv2.imshow("gaussian",img)
            else:
                k=2*val+1
                gaussian = cv2.GaussianBlur(img,(k,k),0)
                cv2.imshow("gaussian",gaussian)
            
        cv2.createTrackbar('Magnitude', 'gaussian', 0, 10,adjust)
        cv2.setTrackbarPos('Magnitude', 'gaussian', 0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process2_2(self):
        self.img_path = filepath1
        self.img = cv2.imread(self.img_path)
        img=self.img.copy()
        cv2.imshow("Bilateral",img)

        def adjust(val):
            if val == 0:
                cv2.imshow("Bilateral",img)
            else:
                k=2*val+1
                gaussian = cv2.bilateralFilter(img,k,90,90)
                cv2.imshow("Bilateral",gaussian)
            
        cv2.createTrackbar('Magnitude', 'Bilateral', 0, 10,adjust)
        cv2.setTrackbarPos('Magnitude', 'Bilateral', 0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process2_3(self):
        self.img_path = filepath1
        self.img = cv2.imread(self.img_path)
        img=self.img.copy()
        cv2.imshow("Median",img)

        def adjust(val):
            if val == 0:
                cv2.imshow("Median",img)
            else:
                k=2*val+1
                gaussian = cv2.medianBlur(img,k)
                cv2.imshow("Median",gaussian)
            
        cv2.createTrackbar('Magnitude', 'Median', 0, 10,adjust)
        cv2.setTrackbarPos('Magnitude', 'Median', 0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def open_file1(self):
        global filepath1
        filepath1=''
        filepath2=''
        filename1, filetype1 = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")                 # start path
        print(filename1, filetype1)
        self.ui.textEdit.setText(filename1)
        filepath1=filename1
        
    def open_file2(self):
        global filepath2
        filename2, filetype2 = QFileDialog.getOpenFileName(self,
                  "Open folder",
                  "./")                 
        print(filename2, filetype2)
        self.ui.textEdit_2.setText(filename2)
        filepath2=filename2



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())