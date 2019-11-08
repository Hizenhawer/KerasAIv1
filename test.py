import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_core.python.keras.models import load_model, Sequential

model = load_model('my_model.h5')
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#image_index = 4444
#plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
#plt.show()
#pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
#print(pred.argmax())
