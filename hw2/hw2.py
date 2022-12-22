from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2
from cv2 import Sobel
import numpy as np
import os
import glob

from hw2_ui import Ui_MainWindow

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton_1.clicked.connect(self.open_forder1)
        self.ui.pushButton_2.clicked.connect(self.open_file1)
        self.ui.pushButton_3.clicked.connect(self.open_file2)
        self.ui.pushButton_4.clicked.connect(self.process1_1)
        self.ui.pushButton_53.clicked.connect(self.process1_2)
        self.ui.pushButton_6.clicked.connect(self.process2_1)
        self.ui.pushButton_7.clicked.connect(self.process2_2)
        self.ui.pushButton_8.clicked.connect(self.process2_3)
        self.ui.pushButton_9.clicked.connect(self.process2_4)
        self.ui.pushButton_10.clicked.connect(self.process2_5)
        self.ui.pushButton_11.clicked.connect(self.process3_1)
        self.ui.pushButton_12.clicked.connect(self.process3_2)
        self.ui.pushButton_13.clicked.connect(self.process4_1)

    #讀取圖片
    def open_forder1(self):

        #讀資料夾
        fordername = QFileDialog.getExistingDirectory(self,
            "Open folder",
            "./")  
          
        self.ui.textEdit_1.setText(fordername)
        self.forderpath=fordername # 用self就能傳給其他方法了

    def open_file1(self):
        filename1, filetype1 = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")            
        self.ui.textEdit_2.setText(filename1)
        self.filepath1=filename1 # 用self就能傳給其他方法了

    def open_file2(self):
        filename2, filetype2 = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")            
        self.ui.textEdit_3.setText(filename2)
        # self.ui.label_t1.setText(filename2)
        self.filepath2=filename2 # 用self就能傳給其他方法了
                
    #影像處理區
    def process1_1(self):
        self.img1 = cv2.imread(self.filepath1)
        img1=self.img1.copy() 
        
        img1_half = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
        img1_gray = cv2.cvtColor(img1_half, cv2.COLOR_BGR2GRAY)
        img1_gaussian = cv2.GaussianBlur(img1_gray,(3,3),0)
        low_threshold = 127
        high_threshold = 255
        img1_edges = cv2.Canny(img1_gaussian, low_threshold, high_threshold)#canny檢測邊緣

        contours1, hierarchy1 = cv2.findContours(img1_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img1_contours = cv2.drawContours(img1_half, contours1, -1, (255,255,0), 1)#第一個參數是要放上contour的圖，二是輪廓來源
        self.contours_num1=str(int(len(contours1)/4))

        self.img2 = cv2.imread(self.filepath2)
        img2=self.img2.copy() 
        
        img2_half = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
        img2_gray = cv2.cvtColor(img2_half, cv2.COLOR_BGR2GRAY)
        img2_gaussian = cv2.GaussianBlur(img2_gray,(3,3),0)
        low_threshold = 127
        high_threshold = 255
        img2_edges = cv2.Canny(img2_gaussian, low_threshold, high_threshold)#canny檢測邊緣

        contours2, hierarchy = cv2.findContours(img2_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img2_contours = cv2.drawContours(img2_half, contours2, -1, (255,255,0), 1)#第一個參數是要放上contour的圖，二是輪廓來源
        self.contours_num2=str(int(len(contours2)/4))

        # cv2.imshow("img_gaussian",img_gaussian)
        # cv2.imshow("img_edges",img_edges)
        cv2.imshow("img1_contours",img1_contours)
        cv2.imshow("img2_contours",img2_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
          
    def process1_2(self):
        self.ui.label_t1.setText("There are "+self.contours_num1+" rings in img1.jpg")
        self.ui.label_t2.setText("There are "+self.contours_num2+" rings in img2.jpg")

    def process2_1(self):

        array_of_img=[]
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        w1=11
        h1=8
        cp_int = np.zeros((w1 * h1, 3), np.float32)
        cp_int[:, :2] = np.mgrid[0:w1, 0:h1].T.reshape(-1, 2)

        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        count=0
        total_count=[]
        for filename in os.listdir(self.forderpath): #可知資料夾中每個檔案名

            img = cv2.imread(self.forderpath + "/" + filename)
            array_of_img.append(img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#轉灰階才能讀參數
            

            ret, cp_img = cv2.findChessboardCorners(img_gray, (w1,h1), None)
            
            corners2 = cv2.cornerSubPix(img_gray,cp_img, (5,5), (-1,-1), criteria)
            obj_points.append(cp_int)
            img_points.append(corners2)
            
            cv2.drawChessboardCorners(img, (w1,h1), cp_img, ret)
            img_half = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            cv2.imshow("2.1",img_half)
            cv2.waitKey(500)#0指無限等待，若輸入1000則為1000毫秒，即1秒

            count+=1
            total_count.append(str(count))
            
        cv2.destroyAllWindows()

        self.ui.comboBox_1.addItems(total_count) #把資料夾內數量加到combobox中

        ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, img_gray.shape[::-1], None, None)
        
        self.Intrinsic=mat_inter
        self.Distortion=coff_dis
        self.R=v_rot
        self.T=v_trans
            
        self.all_imgs=array_of_img

    def process2_2(self):
        print("Intrinsic:\n",self.Intrinsic)

    def process2_3(self):
        index=self.ui.comboBox_1.currentIndex()+1

        r=cv2.Rodrigues(self.R[index])[0]#會有一個3x3矩陣跟3x9Jacobin
        t=self.T[index]
        ex=np.c_[r,t]#旋轉加平移
        print("Extrinsic:\n",ex)

    def process2_4(self):
        print("Distortion:\n",self.Distortion)

    def process2_5(self):
        for filename in os.listdir(self.forderpath): #可知資料夾中每個檔案名

            img = cv2.imread(self.forderpath + "/" + filename)
        
            img_undistorted=cv2.undistort(img,self.Intrinsic,self.Distortion)
            imgs = np.hstack([img,img_undistorted])#直接把矩陣疊在一起輸出兩張
            img_okhalf = cv2.resize(imgs, (0, 0), fx=0.25, fy=0.25)
            cv2.imshow("2.5",img_okhalf)
            cv2.waitKey(500)#0指無限等待，若輸入1000則為1000毫秒，即1秒


        cv2.destroyAllWindows()

    def process3_1(self):
        word=self.ui.lineEdit.text() #抓輸入的字
        images = glob.glob(self.forderpath+"\*.bmp")
        #Calibration
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        w1=11
        h1=8
        cp_int = np.zeros((w1 * h1, 3), np.float32)
        cp_int[:, :2] = np.mgrid[0:w1, 0:h1].T.reshape(-1, 2)

        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        for img_o in images:
            img = cv2.imread(img_o)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#轉灰階才能讀參數
            
            ret, cp_img = cv2.findChessboardCorners(img_gray, (w1,h1), None)
            corners2 = cv2.cornerSubPix(img_gray,cp_img, (5,5), (-1,-1), criteria)
            obj_points.append(cp_int)
            img_points.append(corners2)

        ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, img_gray.shape[::-1], None, None)

        fs = cv2.FileStorage('./Q3_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        #取得所有字母座標
        x=7
        y=5
        number=0
        total_point=[]
        for c in word:
            number+=1
            x_c=number%3
            ch = fs.getNode(c).mat()
            point=(np.float32(ch).reshape(-1,3))

            if number <= 3:
                #做位移
                for i in point:
                    i[1]+=5
                    if x_c == 1:
                        i[0]+=7
                    elif x_c == 2:
                        i[0]+=4
                    elif x_c == 0:
                        i[0]+=1
                    
            elif number > 3 & number <= 6:
                for i in point:
                    i[1]+=2
                    if x_c == 1:
                        i[0]+=7
                    elif x_c == 2:
                        i[0]+=4
                    elif x_c == 0:
                        i[0]+=1
            else:
                print("Wrong!!!")
                
            total_point.append(point)

        #拿轉換後的2D點並在image上畫線
        count=0
        for img_o in images:
            img = cv2.imread(img_o)
            image=img.copy()
            for point in total_point:
                #Projection得2D點
                imgpts, jac = cv2.projectPoints(point, v_rot[count], v_trans[count], mat_inter, coff_dis)
                crd=np.uint(imgpts).reshape(-1,2,2)

                #畫線
                for i in range(len(crd)):
                    image = cv2.line(image, tuple(crd[i][0]), tuple(crd[i][1]), (0, 0, 255), 10)

            img_half = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
            count+=1
            cv2.imshow('test', img_half)
            cv2.waitKey(1000)

        cv2.destroyAllWindows()    

    def process3_2(self):
       
        word=self.ui.lineEdit.text() #抓輸入的字
        images = glob.glob(self.forderpath+"\*.bmp")
        #Calibration
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        w1=11
        h1=8
        cp_int = np.zeros((w1 * h1, 3), np.float32)
        cp_int[:, :2] = np.mgrid[0:w1, 0:h1].T.reshape(-1, 2)

        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane.

        for img_o in images:
            img = cv2.imread(img_o)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#轉灰階才能讀參數
            
            ret, cp_img = cv2.findChessboardCorners(img_gray, (w1,h1), None)
            corners2 = cv2.cornerSubPix(img_gray,cp_img, (5,5), (-1,-1), criteria)
            obj_points.append(cp_int)
            img_points.append(corners2)

        ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, img_gray.shape[::-1], None, None)

        fs = cv2.FileStorage('./Q3_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        #取得所有字母座標
        x=7
        y=5
        number=0
        total_point=[]
        for c in word:
            number+=1
            x_c=number%3
            ch = fs.getNode(c).mat()
            point=(np.float32(ch).reshape(-1,3))

            if number <= 3:
                #做位移
                for i in point:
                    i[1]+=5
                    if x_c == 1:
                        i[0]+=7
                    elif x_c == 2:
                        i[0]+=4
                    elif x_c == 0:
                        i[0]+=1
                    
            elif number > 3 & number <= 6:
                for i in point:
                    i[1]+=2
                    if x_c == 1:
                        i[0]+=7
                    elif x_c == 2:
                        i[0]+=4
                    elif x_c == 0:
                        i[0]+=1
            else:
                print("Wrong!!!")
                
            total_point.append(point)

        #拿轉換後的2D點並在image上畫線
        count=0
        for img_o in images:
            img = cv2.imread(img_o)
            image=img.copy()
            for point in total_point:
                #Projection得2D點
                imgpts, jac = cv2.projectPoints(point, v_rot[count], v_trans[count], mat_inter, coff_dis)
                crd=np.uint(imgpts).reshape(-1,2,2)

                #畫線
                for i in range(len(crd)):
                    image = cv2.line(image, tuple(crd[i][0]), tuple(crd[i][1]), (0, 0, 255), 10)

            img_half = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
            count+=1
            cv2.imshow('test', img_half)
            cv2.waitKey(1000)

        cv2.destroyAllWindows()    

    def process4_1(self):
       
        imgL = cv2.imread(self.filepath1,0)
        imgR = cv2.imread(self.filepath2,0)


        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL, imgR)
        disp=disparity
        disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity = cv2.resize(disparity, (0, 0), fx=0.25, fy=0.25)

        imgL_disparity = cv2.resize(imgL, (0, 0), fx=0.25, fy=0.25)
        imgR_disparity = cv2.resize(imgR, (0, 0), fx=0.25, fy=0.25)

        cv2.imshow('image', disparity)
        # cv2.imshow('imgL', imgL_disparity)
        # cv2.imshow('imgR', imgR_disparity)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        imgL = cv2.imread(self.filepath1)
        imgR = cv2.imread(self.filepath2)
        imgL_disparity = cv2.resize(imgL, (0, 0), fx=0.25, fy=0.25)
        imgR_disparity = cv2.resize(imgR, (0, 0), fx=0.25, fy=0.25)


        def draw(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                imgL=imgL_disparity.copy()
                imgR_C=imgR.copy()
                cv2.circle(imgL, (x,y), 1, (0,0,255), 7)
                if disparity[y][x]!=0:
                    cv2.circle(imgR_C, (4*x-(disparity[y][x]),y*4), 1, (0,0,255), 28)
                imgR_r = cv2.resize(imgR_C, (0, 0), fx=0.25, fy=0.25)
                cv2.imshow('imgL', imgL)
                cv2.imshow('imgR', imgR_r)
                # print(disp[y][x])


        cv2.imshow('imgL', imgL_disparity)
        cv2.imshow('imgR', imgR_disparity)
        cv2.setMouseCallback('imgL', draw)


        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec_())