import os
import cv2
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.models import Model
train_images = []
person_name = []
test_images=[]
p_name=[]
datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)

folder = 'Augment2'
folder_train = os.path.join(folder,'Train Set')

folder_test = os.path.join(folder,'Test Set')
for person in os.listdir(folder_train):
    person_folder = os.path.join(folder_train,person)
    for ear in os.listdir(person_folder):
        ear_image = os.path.join(person_folder,ear)
        if ".png" in ear_image:
            img = cv2.imread(ear_image)
            img = cv2.resize(img,(224,224))
            train_images.append(img)
            person_name.append(int(person))           
print("Loaded all images available in Training Set") 

train_images = np.array(train_images)
person_name = np.array(person_name)

for person in os.listdir(folder_test):
    person_folder = os.path.join(folder_train,person)
    for ear in os.listdir(person_folder):
        ear_image = os.path.join(person_folder,ear)
        if ".png" in ear_image:
            img = cv2.imread(ear_image)
            img = cv2.resize(img,(224,224))
            test_images.append(img)
            p_name.append(int(person))           
print("Loaded all images available in Testing Set") 
x_test=test_images
y_test=p_name

x_train, x_valid, y_train, y_valid = train_test_split(train_images, person_name, test_size=0.2, random_state=0)
#print(y_train)
#print(len(y_train))

#Converting to binary class
y_train = keras.utils.to_categorical(y_train, 151)
y_valid = keras.utils.to_categorical(y_valid, 151)

y_train = y_train[:,0:150]
y_valid = y_valid[:,0:150]

#Creating ResNet model
f_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3),  pooling=None)
x=f_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
predictions = keras.layers.Dense(150,activation='softmax')(x)

model = Model(f_model.input,predictions)
#model.summary()
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x_train, y_train,batch_size=70,epochs=1,verbose=1)



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=70,
          epochs=1,
          verbose=1,
          validation_data=(x_valid, y_valid))

results = np.argmax(model.predict(x_test),axis = 1)
results = np.reshape(results,(len(results),1))

#accuracy 

print("Accuracy: ",(sum(results == y_test))/len(results) * 100)