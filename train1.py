import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

#unpack the data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#x_train = tf.keras.utils.normalize(x_train, axis= 1)
#x_test = tf.keras.utils.normalize(x_train, axis= 1)


#this is model
model = tf.keras.Sequential()
#input layer
model.add(tf.keras.layers.Flatten())
#this is nuarle network
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#this is out put layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
#out put laye, we must specify how many clasifications are there

#now our model is over now we have to train the model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss,val_acc)


model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predic = new_model.predict([x_test])
print(predic)
# plt.imshow(x_train[0], cmap=plt.cm.binary)
# #cmap is color map , binary is black and white
# plt.show()
# print(x_train[0])

print(np.argmax(predic[0]))
plt.imshow(x_test[0])
plt.show()


