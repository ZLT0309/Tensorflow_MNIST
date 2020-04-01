# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import seaborn as sn 
np.random.seed(10)

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

print(tf.__version__)
print(tf.keras.__version__)
print(cv2.__version__)

(X_train_image,y_train_label),(X_test_image,y_test_label)=mnist.load_data()

print('train data=',len(X_train_image))
print(' test data=',len(X_test_image))
print('train data=',X_train_image.shape)
print(' test data=',X_test_image.shape)

def plot_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap=plt.cm.binary)
    plt.show()

#plot_image(X_train_image[1])
#print(y_train_label[1])

x_Train=X_train_image.reshape(60000,784).astype('float32')
x_Test =X_test_image.reshape(10000,784).astype('float32')

x_Train_normalize=x_Train/255
x_Test_normalize=x_Test/255

y_TrainOneHot=to_categorical(y_train_label,10)
y_TestOneHot=to_categorical(y_test_label,10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
model=Sequential()
#model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=512,input_dim=784,kernel_initializer='normal', kernel_regularizer=l2(0.003),activation='relu'))
model.add(Dropout(0.3))
#model.add(Dense(units=128,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=512,input_dim=784,kernel_initializer='normal', kernel_regularizer=l2(0.003),activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax')) 
print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x=x_Train_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)

scores=model.evaluate(x_Test_normalize,y_TestOneHot)

#print ('accuracy=',scores[1])
print('accuracy=',scores)
prediction=model.predict_classes(x_Test)
pred=model.predict(x_Test)

pd.crosstab(y_test_label,prediction,rownames=['labels'],colnames=['predict'])
df=pd.DataFrame({'label':y_test_label,'predict':prediction})
'''
    把原始的数据其标签和实际预测不同的给标记出来,绘制交叉混淆矩阵图形
'''
#把原始的数据其标签和实际预测不同的给标记出来，同时记录在林大贵著的书中
n_classes=10
def DrawingConfusionMatrix(y_test,scnn_predicted,n_classes):
    #scnn_cm = confusion_matrix(np.argmax(y_test, axis=1), scnn_predicted)
    scnn_cm = confusion_matrix(y_test,scnn_predicted)
    scnn_df_cm = pd.DataFrame(scnn_cm, range(n_classes), range(n_classes))
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4) 
    sn.heatmap(scnn_df_cm, annot=True, annot_kws={"size": 12}) # font size
    plt.show()
    return

#DrawingConfusionMatrix(y_test_label, prediction,n_classes)

'''
    测试方案第1种：预测集中共10张图片，把这10张图片读进来，然后组织成一个数组
'''
def imgtobin(im):
    for i in range(0,27):
        for j in range(0,27):
            if im[i][j]>128:
                im[i][j]=0
            else:
                im[i][j]=255
    return im

imaglist=[]
idx=0
for i in range(10):
    filename='d:\\ims\\%s.png'%idx
    res=cv2.resize(cv2.imread(filename,0),(28,28))
    res=imgtobin(res).reshape(-1)
    imaglist.append(res)
    idx+=1

testimg=np.array(imaglist)
testing=testimg.astype('float32')
testingNorm=testing/255.0
result=model.predict_classes(testingNorm)
print(result)

sum = 0
for i in range(0,len(result)):
    if result[i] == i:
        sum += 1
sum /= len(result)

print("sum=", sum)


