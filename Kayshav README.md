from flask import Flask, render_template, flash, request, session,send_file
from flask import render_template, redirect, url_for, request
import warnings
import datetime
import cv2
app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
@app.route("/")
def homepage():
    return render_template('index.html')
@app.route("/Training")
def Training():
    return render_template('Tranning.html')
@app.route("/Test")
def Test():
    return render_template('Test.html')
@app.route("/train", methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        import model as model
        return render_template('Tranning.html')
@app.route("/testimage", methods=['GET', 'POST'])
def testimage():
    if request.method == 'POST':
        file = request.files['fileupload']
        file.save('static/Out/Test.jpg')
        img = cv2.imread('static/Out/Test.jpg')
        if img is None:
            print('no data')
        img1 = cv2.imread('static/Out/Test.jpg')
        print(img.shape)
        img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
        original = img.copy()
        neworiginal = img.copy()
        cv2.imshow('original', img1)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1S = cv2.resize(img1, (960, 540))
        cv2.imshow('Original image', img1S)
        grayS = cv2.resize(gray, (960, 540))
        cv2.imshow('Gray image', grayS)
        gry = 'static/Out/gry.jpg'
        cv2.imwrite(gry, grayS)
        from PIL import  ImageOps,Image
        im = Image.open(file)
        im_invert = ImageOps.invert(im)
        inv = 'static/Out/inv.jpg'
        im_invert.save(inv, quality=95)
        dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
        cv2.imshow("Nosie Removal", dst)
        noi = 'static/Out/noi.jpg'
        cv2.imwrite(noi, dst)
        import warnings
        warnings.filterwarnings('ignore')
        import tensorflow as tf
        classifierLoad = tf.keras.models.load_model('firemodel.h5')
        import numpy as np
        from keras.preprocessing import image
        test_image = image.load_img('static/Out/Test.jpg', target_size=(200, 200))
        img1 = cv2.imread('static/Out/Test.jpg')
        # test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)
        out = ''
        pre = ''
        if result[0][0] == 1:
            out = "Cyclone"
        elif result[0][1] == 1:
            out = "Earthquake"
        elif result[0][2] == 1:
            out = "Flood"
        elif result[0][3] == 1:
            out = "Wildfire"
        org = 'static/Out/Test.jpg'
        gry ='static/Out/gry.jpg'
        inv = 'static/Out/inv.jpg'
        noi = 'static/Out/noi.jpg'
        return render_template('Test.html',result=out,org=org,gry=gry,inv=inv,noi=noi)
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
# Part 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
batch_size = 32
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'Data',  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 200 x 200
        batch_size=batch_size,
        # Specify the classes explicitly
        classes = ['Cyclone','Earthquake','Flood','Wildfire'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
import tensorflow as tf
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()
from tensorflow.keras.optimizers import RMSprop
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
total_sample=train_generator.n
n_epochs = 10
history = model.fit_generator(
        train_generator,
        steps_per_epoch=int(total_sample/batch_size),
        epochs=n_epochs,
        verbose=1)
model.save('firemodel.h5')
acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
# Train and validation accuracy
plt.plot(epochs, acc, 'b', label=' accurarcy')
plt.title('  accurarcy')
plt.legend()
plt.figure()
# Train and validation loss
plt.plot(epochs, loss, 'b', label=' loss')
plt.title('  loss')
plt.legend()
plt.show()
#!C:\Users\Fantasy-PC\PycharmProjects\KerasCnn\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'pip==19.0.3','console_scripts','pip3.7'
__requires__ = 'pip==19.0.3'
import re
import sys
from pkg_resources import load_entry_point
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('pip==19.0.3', 'console_scripts', 'pip3.7')()
    )

