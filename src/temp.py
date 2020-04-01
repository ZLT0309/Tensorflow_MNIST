import tensorflow as tf
from tensorflow import keras
#import librosa
from tensorflow.keras.datasets import mnist, fashion_mnist
import cv2
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)
print(librosa.__version__)
print(cv2.__version__)


(x_train,y_train),(x_test,y_test) = mnist.load_data()
(xf_train,yf_train),(xf_test,yf_test) = fashion_mnist.load_data()

print(x_train.shape)
#print(x_train)
#print(xf_train)


#img = cv2.imread("d:\\timg.jpg",1)
#plt.imshow(img,cmap='gray')
#plt.show()

plt.subplot(221)
#plt.imshow(x_train[0],cmap=plt.get_cmap('gray'))
plt.imshow(x_train[0],cmap='gray')
plt.subplot(222)
print(y_train[0])
plt.imshow(x_train[1],cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2],cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3],cmap=plt.get_cmap('gray'))
plt.show()
