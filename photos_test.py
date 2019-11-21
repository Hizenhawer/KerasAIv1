import glob

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from tensorflow_core.python.keras.models import load_model

import globals

model = load_model(globals.MODEL_PATH, custom_objects={'softmax_v2': tf.nn.softmax})


def test(path):
    image_list = []
    for filename in glob.glob(path):
        im = image.load_img(path=filename, color_mode="grayscale", target_size=(28, 28, 1))
        im = image.img_to_array(im)
        im = im.astype('float32')
        im /= 255
        im *= (-1)
        im += 1
        #im = (im > 0.4) * 1.0 + ((im > 0.28) & (im <= 0.29999)) * 0.5
        image_list.append(im)
    predict(image_list)


def predict(image_list):
    for photo in image_list:
        plt.imshow(photo.reshape(28, 28), cmap='Greys')
        plt.draw()
        pred = model.predict(photo.reshape(1, 28, 28, 1))
        print(pred.argmax())
        plt.show()


test(globals.TEST_2_PATH)
