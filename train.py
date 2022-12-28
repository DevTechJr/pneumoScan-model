# import matplotlib.pyplot as plt
# import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.models import load_model
import cv2
# from cv2 import cv2
import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow


labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train = get_training_data('chest_xray/train')
test = get_training_data('chest_xray/test')
val = get_training_data('chest_xray/val')

print("prepared data")

# Prepare Data for use

x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

positives=[]
negatives=[]
for index in range(len(y_train)):
    if y_train[index]:
        positives.append(x_train[index])
    else:
        negatives.append(x_train[index])

print("done positives and negatives sort")

# To view case amounts, graphically
# numNegatives = len(negatives)
# numPositives = len(positives)
# plt.bar(labels,[numNegatives,numPositives],color=["red","blue"])
# plt.title("Number Of Cases From Training Data")
# plt.ylabel("Count of cases")
# plt.xlabel("Class Types")

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)
x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)
x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

print("trained data in np arrays")
# With data augmentation to prevent overfitting and handling the imbalance in dataset

datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range = 30,
        zoom_range = 0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip = True,
        vertical_flip=False)

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))

model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
 # type: ignoreprint("starting epochs")
history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 12 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])


print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")



def predict(img):
    img = np.array(img)/255
    img = img.reshape(-1,150,150,1)
    isSick = model.predict(img)
    resultdata = str(isSick[0])
    resultdata = float(resultdata)
#     print(type(result))
    if resultdata < 0.5:
        result = "Normal"
#         diagnosis = f"Percentage is {str(result*100)}"
        diagnosis = f"Diagnosis Percentage Is {str(resultdata*100)}%. You are healthy!"
    else:
        result = "Pneumonic"
        diagnosis = f"Diagnosis Percentage Is {str(resultdata*100)}%.\n\
        You are suffering from Pneumonia and should seek immediate medical assistance."
    return result, diagnosis

uploaded_file = st.file_uploader("Choose a file")

result, diagnosis = predict(uploaded_file)

# submitBtn = st.button("Get Diagnosis", on_click=getDiagnosis,disabled=False)

print(result)
print(diagnosis)

model.save('pneumo2')




