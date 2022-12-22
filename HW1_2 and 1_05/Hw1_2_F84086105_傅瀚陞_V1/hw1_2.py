from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
from cv2 import Sobel
import numpy as np

from hw1_2_ui import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton.clicked.connect(self.open_file1) 
        self.ui.pushButton_1.clicked.connect(self.process3_1)
        self.ui.pushButton_2.clicked.connect(self.process3_2)
        self.ui.pushButton_3.clicked.connect(self.process3_3)
        self.ui.pushButton_4.clicked.connect(self.process3_4)
        self.ui.pushButton_5.clicked.connect(self.process4_1)
        self.ui.pushButton_6.clicked.connect(self.process4_2)
        self.ui.pushButton_7.clicked.connect(self.process4_3)
        self.ui.pushButton_8.clicked.connect(self.process4_4)

    #讀取圖片
    def open_file1(self):
        filename1, filetype1 = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")            
        self.ui.textEdit.setText(filename1)
        self.filepath=filename1 # 用self就能傳給其他方法了
                
    #影像處理區
    def process3_1(self):
        self.img = cv2.imread(self.filepath)
        img=self.img.copy() 
        img_gray_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        filter1 = np.array([
                        [0.045,0.122,0.045],
                        [0.122,0.332,0.122],
                        [0.045,0.122,0.045]
                    ])
        n,m = img_gray_1.shape
        img_new = []
        for i in range(n-3):
            line = []
            for j in range(m-3):
                mat = img_gray_1[i:i+3,j:j+3]
                line.append(np.sum(np.multiply(mat, filter1)))
            img_new.append(line)
        Gaussian_Blur =np.array(img_new,dtype=np.uint8)
        global gaussian
        gaussian=Gaussian_Blur.copy()
        cv2.imshow("Gaussian Blur",Gaussian_Blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
          
    def process3_2(self):
        filter2 = np.array([
                [-1,0,1],
                [-2,0,2],
                [-1,0,1]
            ])
        n,m = gaussian.shape
        img_new = []
        for i in range(n-3):
            line = []
            for j in range(m-3):
                mat = gaussian[i:i+3,j:j+3]
                point=np.sum(np.multiply(mat, filter2))
                if point > 255:
                    line.append(255)
                elif point < 0:
                    line.append(0)
                else :
                    line.append(point)
            img_new.append(line)
        Sobel_X =np.array(img_new,dtype=np.uint8)
        print(Sobel_X.shape)
        global s_x
        s_x=Sobel_X.copy()
        cv2.imshow("Sobel_X",Sobel_X)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process3_3(self):
        filter3 = np.array([
                        [-1,-2,-1],
                        [0,0,0],
                        [1,2,1]
                    ])
        n,m = gaussian.shape
        img_new = []
        for i in range(n-3):
            line = []
            for j in range(m-3):
                mat = gaussian[i:i+3,j:j+3]
                point=np.sum(np.multiply(mat, filter3))
                if point > 255:
                    line.append(255)
                elif point < 0:
                    line.append(0)
                else :
                    line.append(point)
            img_new.append(line)
        Sobel_Y =np.array(img_new,dtype=np.uint8)
        print(Sobel_Y.shape)
        global s_y
        s_y=Sobel_Y.copy()
        cv2.imshow("Sobel_Y",Sobel_Y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process3_4(self):
        n,m=s_x.shape
        new_img=[]
        for i in range(n):
            line = []
            for j in range(m):
                x=s_x[i][j]
                y=s_y[i][j]
                ma=(x**2+y**2)**0.5
                mag=ma/361*255
                line.append(mag)
            new_img.append(line)
        Magnitude =np.array(new_img,dtype=np.uint8)
        cv2.imshow("Magnitude",Magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process4_1(self):

        self.img = cv2.imread(self.filepath)
        img=self.img.copy()
        img_resize = cv2.resize(img, (215, 215))
        M = np.float32([[1, 0, 0], [0, 1, 0]])
        output = cv2.warpAffine(img_resize, M, (430, 430))
        global image4_1
        image4_1=img_resize.copy()
        cv2.imshow("Resized Image",output)
        cv2.resizeWindow("Resized Image", 430, 430)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process4_2(self):

        M_1 = np.float32([[1, 0, 0], [0, 1, 0]])
        output_1 = cv2.warpAffine(image4_1, M_1, (430, 430))
        M_2 = np.float32([[1, 0, 215], [0, 1, 215]])
        output_2 = cv2.warpAffine(image4_1, M_2, (430, 430))
        output_3=cv2.addWeighted(output_1,1,output_2,1,0)
        global image4_2
        image4_2=output_3.copy()
        cv2.imshow("Translation Image",output_3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process4_3(self):
        R = cv2.getRotationMatrix2D((215, 215), 45, 0.5)
        output = cv2.warpAffine(image4_2, R, (430, 430))
        global image4_3
        image4_3=output.copy()
        cv2.imshow("Rotation Image",output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process4_4(self):
       
        p1 = np.float32([[50,50],[200,50],[50,200]])
        p2 = np.float32([[10,100],[100,50],[100,250]]) 
        S = cv2.getAffineTransform(p1, p2)
        output = cv2.warpAffine(image4_3, S, (430, 430))
        cv2.imshow("Shear Image",output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())