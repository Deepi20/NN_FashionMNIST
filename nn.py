mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels),(testing_images,testing_labels) = mnist.load_data()
import matplotlib.pyplot as plt
print(training_labels[0])
print(training_images[0])
training_images = training_images/255.0
testing_images = testing_images/255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128,activation = tf.nn.relu),
                                   tf.keras.layers.Dense(10,activation = tf.nn.softmax)])
model.compile(optimizer = tf.keras.optimizers.Adam(),
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
model.fit(training_images,training_labels,epochs = 100)
