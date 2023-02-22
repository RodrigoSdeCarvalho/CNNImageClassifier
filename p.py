# %%
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ASSETS_PATH = os.path.join(os.getcwd(), 'Assets')

# %%
def extract_zip_folder(zip_name:str) -> None:
    """Extracts a zip folder to a given path.
    
    Args:
        zip_folder_path (str): name of the zip folder.
    """
    zip_path = os.path.join(ASSETS_PATH, zip_name)
    zip = zipfile.ZipFile(file=zip_path, mode = 'r')
    zip.extractall(os.path.join(ASSETS_PATH))
    zip.close()

# %%
extract_zip_folder("homer_bart.zip")

# %%
# Test if the images are extracted correctly

tf.keras.preprocessing.image.load_img(os.path.join(ASSETS_PATH, "homer_bart", "training_set", "bart", "bart29.bmp"))

# %%
# Objct to create image training sets.

training_generator = ImageDataGenerator(ImageDataGenerator(rescale=1./255,
                                         rotation_range=7,
                                         horizontal_flip=True,
                                         zoom_range=0.2))

# Classes are the folders in the training set.

training_dataset = training_generator.flow_from_directory(os.path.join(ASSETS_PATH, "homer_bart", "training_set"),
                                                              target_size = (64, 64),
                                                              batch_size = 8,
                                                              class_mode = 'categorical',
                                                              shuffle = True)

# %%
# Test set. No need to apply transformations.

gerador_teste = ImageDataGenerator(rescale=1./255)
test_dataset = gerador_teste.flow_from_directory(os.path.join(ASSETS_PATH, "homer_bart", "test_set"),
                                                  target_size = (64,64),
                                                  batch_size = 8,
                                                  class_mode = 'categorical',
                                                  shuffle=False)

# %%
def compile_CNN_classifier() -> Sequential:
    """Defines the layers of the classifier and compiles it.
    
    returns: Sequential object with the classifier.
    """
    CNN_classifier = Sequential()

    # First convolutional layer
    CNN_classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64, 64, 3)))
    CNN_classifier.add(MaxPool2D(pool_size=(2,2)))

    # Second convolutional layer
    CNN_classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    CNN_classifier.add(MaxPool2D(pool_size=(2,2)))

    # Third convolutional layer
    CNN_classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    CNN_classifier.add(MaxPool2D(pool_size=(2,2)))

    # Flattening
    CNN_classifier.add(Flatten())

    # Fully connected layer
    CNN_classifier.add(Dense(units = 577, activation='relu'))
    CNN_classifier.add(Dense(units = 577, activation='relu'))
    
    # Output layer. Softmax calculates the probability of each neuron (class)
    CNN_classifier.add(Dense(units = 2, activation='softmax'))

    # Compiling the neural network
    CNN_classifier.summary()
    CNN_classifier.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return CNN_classifier

# %%
from os.path import exists
from keras.models import load_model
from keras.callbacks import EarlyStopping

def get_classifier() -> Sequential:
    """Loads the classifier if it exists, or creates and trains a new one.
    
    returns: Sequential object with the classifier.
    """
    if exists(os.path.join(ASSETS_PATH, 'classifier_model.h5')):
        classifier = load_model(os.path.join(ASSETS_PATH, 'classifier_model.h5'))
    else:
        classifier = compile_CNN_classifier()

        # Trains the classifier
        classifier.fit(training_dataset, epochs=50)
        
        # Saves the classifier
        classifier.save(os.path.join(ASSETS_PATH, "classifier_model.h5"))

    return classifier

# %%
# Classifier evaluation

classifier = get_classifier()

predictions = classifier.predict(test_dataset)
print(predictions)
predictions = np.argmax(predictions, axis=1)

print(accuracy_score(test_dataset.classes, predictions))
cm = confusion_matrix(test_dataset.classes, predictions)
print(classification_report(test_dataset.classes, predictions))
sns.heatmap(cm, annot=True)



# %%
# Classifying a new image

image = cv2.imread(os.path.join(ASSETS_PATH, "homer_bart", "test_set", "homer",'homer15.bmp'))
print(image)
image = cv2.resize(image, (64, 64))
image = image / 255
image = image.reshape(-1, 64, 64, 3)
prediction = classifier.predict(image)
prediction = np.argmax(prediction, axis=1)

if prediction == 0:
  print('Bart')
else:
  print('Homer')



