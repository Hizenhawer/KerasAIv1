import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_core.python.keras.models import load_model, Sequential
import globals

model = load_model(globals.MODEL_PATH,custom_objects={'softmax_v2': tf.nn.softmax})
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.draw()
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
plt.show()

image_index = 4443
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.draw()
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
plt.show()

image_index = 4442
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.draw()
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
plt.show()
