import numpy as np # linear algebra
import json
import cv2
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import SVC
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

clf=joblib.load("train_model.m")
face_cascade=cv2.CascadeClassifier('Haarcascades_Datasets/haarcascade_frontalface_default.xml')#copy the locations
img_length=80
ppc=16

# img = cv2.imread("test/"+"test6.png")


def GETimg(StringPath):#define the path here
    img=cv2.imread(StringPath)
    return img

def predGrayCropped(img):
    if type(img) is np.ndarray:
        #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rm = cv2.resize(img, (img_length, img_length))

    #     data_gray = color.rgb2gray(rm)
    data_gray = rm
    fd, hog_image = hog(data_gray, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2',
                        visualise=True)
    hog_feature = np.array(fd)
    y = clf.predict(hog_feature.reshape(1, -1))  # 此处test_X为特征集
    ######y[o]是预测结果 the result is y[0]
    # -----区间等于y[0]

    result1 = "predicted age is :" + str(y[0] * 10) + " ~ " + str(y[0] * 10 + 9)
    print(result1)

    result = y[0]
    return (result)


def AgePredictInWholePicuture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # in face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 218, 185), 3)

        # extracting the facial part
        roi_gray = gray[y:y + h, x:x + w]

        # roi_color = img[y:y + h, x:x + w]
        # simg = cv2.resize(roi_gray, (80, 80))
    ageP = predGrayCropped(roi_gray)
    cv2.putText(img, "%s" % str(ageP * 10) + "~" + str(ageP * 10 + 9), (x - 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


    pass


# def AgePredictInWholePicutureReal(img):  # real-time from camera 没有BGR2RGB
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         # in face
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 218, 185), 3)
#
#         # extracting the facial part
#         roi_gray = gray[y:y + h, x:x + w]
#
#         # roi_color = img[y:y + h, x:x + w]
#         # simg = cv2.resize(roi_gray, (80, 80))
#     ageP = predGrayCropped(roi_gray)
#     cv2.putText(img, "%s" % str(ageP * 10) + "~" + str(ageP * 10 + 9), (x - 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 (0, 255, 255), 1)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # plt.imshow(img)
#     # plt.show()
#
#     return (img)


#     return (roi_gray)
img=GETimg("test/"+"test3.png")
AgePredictInWholePicuture(img)

