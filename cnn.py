import tensorflow as tf
import numpy as np
from tensorflow import keras
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels),(testing_images,testing_labels) = mnist.load_data()
import matplotlib.pyplot as plt
print(training_labels[0])
print(training_images[0])
training_images = training_images/255.0
testing_images = testing_images/255.0
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.91):
      print("\nReached 91% accuracy so cancelling training!")
      self.model.stop_training = True

        
callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation ='relu',input_shape = (28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(10,activation = 'softmax')
])
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.summary()
model.fit(training_images,training_labels,epochs = 5, callbacks = [callbacks])
test_loss = model.evaluate(test_images,test_labels)
