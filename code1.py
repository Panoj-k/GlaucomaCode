#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from skimage import io
import keras
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, Sequential, load_model
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from tensorflow.keras import optimizers
import tensorflow as tf

#%%
#Get picture and
directory = '.\Picture'
n1 = 0
n2 = 0
n3 = 0
y = []
X = []
for filename in os.listdir(directory):
  print('______________________________________________________')
  print(filename)
  n = 0
  print(os.path.join(directory,filename))
  for image_file in os.listdir(os.path.join(directory,filename)):
    print(image_file)
    try:
      if n >= 30:
        break
      n +=1
      image = cv2.imread(os.path.join(directory,filename,image_file))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.resize(image,(1920, 2560),interpolation = cv2.INTER_AREA)
      X.append(image)

      if filename == 'Normal':
        y.append(0)
        n1+=1
      elif filename == 'Glaucoma':
        y.append(2)
        n2+=1
      else:
        y.append(1)
        n3+=1
    except:
      continue

y = np.array(y)
X = np.array(X)
print(X.shape,y.shape)
print('No. normal: ',n1,'No. suspected: ',n2,'No. glaucoma: ',n3)

# %%
def splitData_class(y, seed_num):
  np.random.seed(seed_num)
  real_train = np.empty([0], dtype= int)
  real_validation = np.empty([0], dtype= int)
  real_test = np.empty([0], dtype= int)
  for i in np.unique(y, axis=0):
    index_random = []
    j = 0
    for e in y:
      if np.array_equal(e,i):
        index_random.append(j)
      j+=1
    index_random = np.array(index_random)
    np.random.shuffle(index_random)
    train_index = index_random[0:round(0.33*index_random.size)]
    validation_index = index_random[round(0.33*index_random.size):round(0.66*index_random.size)]
    test_index = index_random[round(0.66*index_random.size):index_random.size]
    real_train = np.append(real_train,train_index)
    real_validation = np.append(real_validation,validation_index)
    real_test = np.append(real_test, test_index)
  return(real_train,real_validation,real_test)

#%%
import keras
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, Sequential, load_model
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from tensorflow.keras import optimizers
import tensorflow as tf

#%%
training_data,validation_data,test_data = splitData_class(y, 5)
X_train=X[training_data]
y_train=y[training_data]
X_test=X[test_data]
y_test=y[test_data]
X_validation=X[validation_data]
y_validation=y[validation_data]
print(X_test.shape)

#%%
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_validation = to_categorical(y_validation)

#%%
def get_custom_model():
    'this function returns model object as an output'
    base_model = InceptionV3(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(300, activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(base_model.input ,output)
    i = 0
    'set some layers to become trainable for transfer learning'
    for layer in base_model.layers:
        layer.trainable=True
        i += 1
    return model
  
#%%
model = get_custom_model()

#%%
model.summary()

#%%
sgd = tf.keras.optimizers.SGD(lr=0.01, clipvalue=0.5)
model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, validation_data = (X_validation, y_validation), batch_size = 1, epochs = 1, shuffle = True, verbose = 2)

#%%
y_pred = model.predict(X_test)
y_pred_round = np.argmax(y_pred, axis = -1)
y_pred_round
#%%
accuracy = y_test == y_pred_round
accuracy_percent = np.sum(accuracy.astype('int'))/(y_test.size)*100
print(accuracy_percent)