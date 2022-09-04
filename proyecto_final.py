# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:21:44 2022

@author: alejandro.ortiz
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from random import shuffle
from sklearn.metrics import confusion_matrix

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
Se utilizo convoluciones
'''

modelo=Sequential()
modelo.add(Convolution2D(32, (3,3),input_shape=(216,384,3),activation='relu'))
modelo.add(MaxPooling2D(pool_size=((2,2))))
modelo.add(Flatten())
modelo.add(Dense(128,activation='relu'))
modelo.add(Dense(50,activation='relu'))
modelo.add(Dense(1,activation='sigmoid'))
modelo.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

x_train=[]
y_train=[]
x_test=[]
y_test=[]

dataTr=[]

import glob
import os
for filename in glob.glob(os.path.join('CarneDataset/train/CLASS_08','*.png')):
    dataTr.append([7,cv2.imread(filename)])
for filename in glob.glob(os.path.join('CarneDataset/train/CLASS_07','*.png')):
    dataTr.append([6,cv2.imread(filename)])
for filename in glob.glob(os.path.join('CarneDataset/train/CLASS_06','*.png')):
    dataTr.append([5,cv2.imread(filename)])
for filename in glob.glob(os.path.join('CarneDataset/train/CLASS_05','*.png')):
    dataTr.append([4,cv2.imread(filename)])
for filename in glob.glob(os.path.join('CarneDataset/train/CLASS_04','*.png')):
    dataTr.append([3,cv2.imread(filename)])
for filename in glob.glob(os.path.join('CarneDataset/train/CLASS_03','*.png')):
    dataTr.append([2,cv2.imread(filename)])
for filename in glob.glob(os.path.join('CarneDataset/train/CLASS_02','*.png')):
    dataTr.append([1,cv2.imread(filename)])
for filename in glob.glob(os.path.join('CarneDataset/train/CLASS_01','*.png')):
    dataTr.append([0,cv2.imread(filename)]) 

shuffle(dataTr)

for i,j in dataTr:
    y_train.append(i)
    x_train.append(j)    
    
x_train=np.array(x_train)
y_train=np.array(y_train)

for filename in glob.glob(os.path.join('CarneDataset/test/CLASS_08','*.png')):
    x_test.append(cv2.imread(filename))
    y_test.append(7)

for filename in glob.glob(os.path.join('CarneDataset/test/CLASS_07','*.png')):
    x_test.append(cv2.imread(filename))
    y_test.append(6)

for filename in glob.glob(os.path.join('CarneDataset/test/CLASS_06','*.png')):
    x_test.append(cv2.imread(filename))
    y_test.append(5)

for filename in glob.glob(os.path.join('CarneDataset/test/CLASS_05','*.png')):
    x_test.append(cv2.imread(filename))
    y_test.append(4) 

for filename in glob.glob(os.path.join('CarneDataset/test/CLASS_04','*.png')):
    x_test.append(cv2.imread(filename))
    y_test.append(3)

for filename in glob.glob(os.path.join('CarneDataset/test/CLASS_03','*.png')):
    x_test.append(cv2.imread(filename))
    y_test.append(2)

for filename in glob.glob(os.path.join('CarneDataset/test/CLASS_02','*.png')):
    x_test.append(cv2.imread(filename))
    y_test.append(1)
    
for filename in glob.glob(os.path.join('CarneDataset/test/CLASS_01','*.png')):
    x_test.append(cv2.imread(filename))
    y_test.append(0)
    
x_test=np.array(x_test)
y_test=np.array(y_test)

modelo.fit(x_train,y_train,batch_size=32,epochs=4,validation_data=(x_test, y_test))

'''
Aquí se escoge la imagen que se quiere realizar el test
'''
nombre_imagen_test = "05-CAPTURE_20220421_053715_321.png"

'''
Aquí se escoge el tipo de clase de jamón de donde queremos comparar
'''
numero_clase_test = 2

ruta_test = "CarneDataset/test/CLASS_0"+ str(numero_clase_test) +"/"+ nombre_imagen_test

I=cv2.imread(ruta_test)

if round(modelo.predict(np.array([I]))[0][0])== (numero_clase_test - 1):
    print("\nJamón de clase: "+ str(numero_clase_test) +"!!\n")
else:
    print("\nno es Jamón de clase "+ str(numero_clase_test) +"!!\n")

x_test1=[]
y_test1=[]

for filename in glob.glob(os.path.join(ruta_test)):
    x_test1.append(cv2.imread(filename))
    y_test1.append(0)

y_pred = modelo.predict(np.array([I]))

'''
Aquí se calcula la matriz de confusión
'''
cm = confusion_matrix(y_test1, y_pred)
print("Matriz de Confusión: \n")
print(cm)
print("\n")

'''
Aquí se elabora el gráfico de la matriz de confusión
'''
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Matriz de Confusión'); 
ax.xaxis.set_ticklabels(['Entrenamiento', 'Prueba']); ax.yaxis.set_ticklabels(['Entrenamiento', 'Prueba']);






