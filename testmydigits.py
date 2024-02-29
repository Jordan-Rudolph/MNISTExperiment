import numpy as np
import matplotlib.pyplot as pp

import tensorflow as tf
from keras import layers

#import imageio as iio
from PIL import Image, ImageChops
import PIL.ImageOps
import os

model = tf.keras.models.load_model('big_brain3')

images = []
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        # check the extension of files
        if file.endswith('.png'):

            rgba_image = PIL.Image.open(file)
            rgba_image = rgba_image.resize((28, 28))
            rgb_image = rgba_image.convert('RGB')
            rgb_image = tf.keras.preprocessing.image.img_to_array(rgb_image)
            rgb_image = tf.image.rgb_to_grayscale(rgb_image)
            rgb_image = tf.squeeze(rgb_image)
            #print(rgb_image.shape)
            #pp.imshow(rgb_image, cmap='gray')
            #pp.show()
            images.append(rgb_image)
#print(len(images))
for i in range(len(images)):
    prediction = model.predict(np.array([images[i]]))
    pp.imshow(images[i], cmap='gray')
    pp.show()
    print("It's a: ", np.argmax(prediction))
