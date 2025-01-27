import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy import interp
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.preprocessing import image
#from image import ImageDataGenerator
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


train_dandelion_dir = os.path.join('train/people')

train_grass_dir = os.path.join('train/crocodile')

valid_dandelion_dir = os.path.join('valid/people')

valid_grass_dir = os.path.join('valid/crocodile')

train_dandelion_names = os.listdir(train_dandelion_dir)

train_grass_names = os.listdir(train_grass_dir)

validation_grass_names = os.listdir(valid_grass_dir)

print('total training dandelion images:', len(os.listdir(train_dandelion_dir)))
print('total training grass images:', len(os.listdir(train_grass_dir)))
print('total validation dandelion images:', len(os.listdir(valid_dandelion_dir)))
print('total validation grass images:', len(os.listdir(valid_grass_dir)))

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_dandelion_pic = [os.path.join(train_dandelion_dir, fname)
                for fname in train_dandelion_names[pic_index-8:pic_index]]
next_grass_pic = [os.path.join(train_grass_dir, fname)
                for fname in train_grass_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_dandelion_pic + next_grass_pic):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
print('a')
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'train/', 
        classes = ['people', 'crocodile'],
        target_size=(200, 200), #(200,200)
        batch_size=2,
        class_mode='binary')
print('b')
validation_generator = validation_datagen.flow_from_directory(
        'valid/', 
        classes = ['people', 'crocodile'],
        target_size=(200, 200),
        batch_size=5,#19
        class_mode='binary',
        shuffle=False)
print('c')
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (200, 200,3)),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
print('d')
model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
print('w')
history = model.fit(train_generator,
      steps_per_epoch=8,
      epochs=2,#15
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)
print('y')
uploaded = ['croc.jpg']

for fn in uploaded:
    path = 'content/' + fn
    img = image.load_img(path, target_size=(200, 200))
    x = image.img_to_array(img)
    plt.imshow(x / 255.)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] < 0.5:
        print(fn + " is a croc")
    else:
        print(fn + " is a person")
