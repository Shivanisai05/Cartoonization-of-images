#installing kaggle
pip install -q kaggle

#uploading the kaggle.json file
from google.colab import files
files.upload()

#making a new directory called kaggle
!mkdir ~/.kaggle

#copying the kaggle.json to the kaggle directory
!cp kaggle.json ~/.kaggle/

#changing the access mode of the file
!chmod 600 ~/.kaggle/kaggle.json

#listing out the kaggle datasets published for compitetions
!kaggle datasets list

#downloading the reauired image dataset from kaggle
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

#manual extraction the data drom the zipfile(in the later of the project i have used keras automated data extraction from the directories and data generation so need to do these things ) 
#all these steps until importing keras are done only for example so need to include these things 
#Unzipping the downloaded file
from zipfile import ZipFile
file_name='chest-xray-pneumonia.zip'
with ZipFile(file_name,'r') as zip_file:
  zip_file.printdir()
  print("Extracting all the files now :)")
  zip_file.extractall()
print("Done")

#listing out the contents of the the unzipped file
import os
os.listdir('chest_xray')

#changing the working directory
os.chdir('chest_xray')
os.getcwd()
os.listdir()

#a sample code to view the image of the downloaded data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
path='train/PNEUMONIA'
Img=[]
for files in os.listdir(path):
  files=path+ '/' +files
  img=Image.open(files)
  img=np.array(img,'uint8')
  Img=img
plt.imshow(Img,cmap='gray')

#extracting the training data
x_train=[]
y_train=[]
for root,dir_,files in os.walk('train'):
  for file in files:
    img_path=os.path.join(root,file)
    identifier=os.path.basename(root)
    
    img=Image.open(img_path)
    img=np.array(img)
    x_train.append(img)
    if identifier=='PNEUMONIA':
      y_train.append(1)
    if identifier=='NORMAL':
      y_train.append(0)

x_train=np.array(x_train)
y_train=np.array(y_train)
plt.imshow(x_train[1],cmap='gray')
print("Total training samples : {}".format(y_train.shape))

#extracting test data
x_test=[]
y_test=[]
for root,dir_,files in os.walk('test'):
  for file in files:
    img_path=os.path.join(root,file)
    identifier=os.path.basename(root)
    
    img=Image.open(img_path)
    img=np.array(img)
    x_test.append(img)
    if identifier=='PNEUMONIA':
      y_test.append(1)
    if identifier=='NORMAL':
      y_test.append(0)

x_test=np.array(x_test)
y_test=np.array(y_test)
plt.imshow(x_test[0],cmap='gray')
print("Total testing samples : {}".format(y_test.shape))

#extracting the validation data
x_val=[]
y_val=[]
for root,dir_,files in os.walk('val'):
  for file in files:
    img_path=os.path.join(root,file)
    identifier=os.path.basename(root)
    
    img=Image.open(img_path)
    img=np.array(img)
    x_val.append(img)
    if identifier=='PNEUMONIA':
      y_val.append(1)
    if identifier=='NORMAL':
      y_val.append(0)

x_val=np.array(x_val)
y_val=np.array(y_val)
plt.imshow(x_val[0],cmap='gray')
print('Validation test size : {}'.format(x_val.shape))

print("Training data size : {}".format(y_train.shape))
print("Testing data size : {}".format(y_test.shape))
print("Validation data size : {}".format(x_val.shape))

#the real things start here
from tensorflow.keras.preprocessing import image as tfk_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend 
import numpy as np
from PIL import Image

image_width,image_height=Image.open('train/PNEUMONIA/person963_bacteria_2888.jpeg').size
print("image size : {} * {}".format(image_width,image_height))

#downscaling the image
image_width=128
image_height=128

train_data_dir='train'
test_data_dir='test'
val_data_dir='val'
num_training_samples=5216
num_testing_samples=624
num_val_samples=16

if backend.image_data_format()=='channels_first':
  input_shape=(3,image_width,image_height)
else:
  input_shape=(image_width,image_height,3)
print("Input shape is {} ".format(input_shape))

train_data_gen=tfk_img.ImageDataGenerator(rescale=1./255,zoom_range=0.3)
test_data_gen=tfk_img.ImageDataGenerator(rescale=1./255,zoom_range=0.2)

train_generator=train_data_gen.flow_from_directory(train_data_dir,target_size=(image_width,image_height),
                                                   batch_size=50,shuffle=True,seed=101,
                                                   classes=['PNEUMONIA','NORMAL'],class_mode='binary')
test_generator=test_data_gen.flow_from_directory(test_data_dir,target_size=(image_width,image_height),
                                                 batch_size=50,shuffle=True,seed=101,
                                                 classes=['PNEUMONIA','NORMAL'],class_mode='binary')

#model developing
#first layer
model=Sequential()
model.add(Conv2D(16,(3,3),strides=(2,2),activation='relu',padding='same',input_shape=(image_width,image_height,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

#second layer
model.add(Conv2D(32,(3,3),strides=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#third layer
model.add(Conv2D(64,(3,3),strides=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#flatten the image
model.add(Flatten())
#add a densely connected layer
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.27))
#output layer
model.add(Dense(units=1,activation='sigmoid'))

model.summary()

#compiling the model
adam_opt=Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy',optimizer=adam_opt,metrics=['accuracy'])

import tensorflow as tf
device_name=tf.test.gpu_device_name()
print(device_name)

#training the model
batch_size=50
model.fit_generator(train_generator,
                    steps_per_epoch=num_training_samples // batch_size,
                    epochs=50,
                    validation_data=test_generator,
                    validation_steps=num_testing_samples//batch_size)

import h5py
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

#authenciate and create PyDrive Client
auth.authenticate_user()
gauth=GoogleAuth()
gauth.credentials=GoogleCredentials.get_application_default()

#save the model
drive=GoogleDrive(gauth)
model.save('dhamo_pneumonia.h5')
model_file=drive.CreateFile({'title':'dhamo_pneumonia.h5'})
model_file.SetContentFile('dhamo_pneumonia.h5')

#download to google drive
drive.CreateFile({'id':model_file.get('id')})

#load the model from drive(drive have to mounted... check  the files option to mount the drive)
from tensorflow.keras.models import load_model
pneumonia_model=load_model('/content/drive/My Drive/dhamo_pneumonia.h5')

#weights of the  features in the model
pneumonia_model.weights

#make predictions with the loaded model from drive
import os
import matplotlib.pyplot as plt
print("MAKING PREDICTIONS...")
for root,dir_,files in os.walk('val'):
  for file in files:
    path=os.path.join(root,file)
    img=tfk_img.load_img(path)
    img=img.resize((128,128))
    img=tfk_img.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    result=pneumonia_model.predict(img)
    pred=''
    if result[0][0]==0:
      pred='PNEUMONIA'
      IMG=Image.open(path)
      IMG=np.array(IMG,'uint8')
      plt.imshow(IMG,cmap='gray')
      plt.title(pred)

    else:
      pred='NORMAL'
      IMG=Image.open(path)
      IMG=np.array(IMG,'uint8')
      plt.imshow(IMG,cmap='gray')
      plt.title(pred)
    plt.show()

#know the accuracy of the model classification
val_gen=tfk_img.ImageDataGenerator(rescale=1./255,zoom_range=0.2)
val_generator=test_data_gen.flow_from_directory('val',target_size=(image_width,image_height),
                                                 batch_size=1,shuffle=False,
                                                 classes=['PNEUMONIA','NORMAL'],class_mode='binary')

pneumonia_model.evaluate_generator(val_generator,steps=100)

print('96% of Pneumonia patient has been correctly identified as PNEUMONIA affected patients')