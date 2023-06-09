# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import os
import cv2

# Load the data
train_data = pd.read_csv('/kaggle/input/extracting-attributes-from-fashion-images-2/train.csv')

# Prepare the data
X = train_data['file_name']  # Image file names
y = train_data['label']  # Class labels
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# Define data augmentation and preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
)
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# Define data augmentation and preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train = pd.DataFrame()
train['file_name'] = X_train 
train['label'] = y_train
# Convert numeric labels to strings
train['label'] = train['label'].astype(str)

# Create the train generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='file_name',
    y_col='label',
    directory='/kaggle/input/extracting-attributes-from-fashion-images-2/train',
    shuffle=True,
    target_size=(100, 100),
    batch_size=32,
    class_mode='sparse'
)
valid = pd.DataFrame()
valid['file_name'] = X_val 
valid['label'] = y_val
# Convert numeric labels to strings
valid['label'] = valid['label'].astype(str)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=valid,
    x_col="file_name",
    y_col="label",
    directory='/kaggle/input/extracting-attributes-from-fashion-images-2/train',
    #subset="validation",  # Use the validation subset of the data
    shuffle=True,
    target_size=(100, 100),  # Update the target size if needed
    batch_size=32,
    class_mode='sparse'
)
# Define the CNN model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)
test = pd.read_csv('/kaggle/input/extracting-attributes-from-fashion-images-2/sample_submission.csv')
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
)
test['label'] = test['label'].astype(str)
test_images = test_generator.flow_from_dataframe(
    dataframe=test,
    x_col='file_name',
    y_col='label',
    directory='/kaggle/input/extracting-attributes-from-fashion-images-2/test',
    target_size=(100,100),
    batch_size=32,
    shuffle=False
)
pred = model.predict(test_images)
pred = np.argmax(pred,axis=1)
sub = pd.read_csv('/kaggle/input/extracting-attributes-from-fashion-images-2/sample_submission.csv')
sub['label'] = pred
sub.to_csv('file1.csv')

