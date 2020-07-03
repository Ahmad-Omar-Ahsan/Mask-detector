from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser(description='To train deep learning model with your mask dataset')
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-p', '--plot', type=str, default='plot.png',help='path to output loss/accuracy plot')
ap.add_argument('-m', '--model', type=str, default='mask_detector.model', help='Path to output face mask detector model')
args = vars(ap.parse_args())

initial_learning_rate = 1e-4
epochs = 20
batch_size = 32

print('loading images....')
image_path = list(paths.list_images(args['dataset']))
data = []
labels = []

for filename in image_path:
    label = filename.split(os.path.sep)[-2]
    
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype='float32')
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

aug= ImageDataGenerator(
    rotation_range = 20,
    zoom_range = 0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)


base_model = MobileNetV2(weights='imagenet',include_top=False, input_tensor=Input(shape=(224,224,3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(128, activation = 'relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation='softmax')(head_model)

model =  Model(inputs= base_model.input, outputs=head_model)

for layer in base_model.layers:
    layer.trainable= False

print('Compiling model....')
opt = Adam(lr= initial_learning_rate, decay = initial_learning_rate/epochs)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

print('training the top layer only..')
H = model.fit(
    aug.flow(train_x, train_y, batch_size=batch_size),
    steps_per_epoch = len(train_x)//batch_size,
    validation_data = (test_x, test_y),
    validation_steps = len(test_x)//batch_size,
    epochs=epochs
)

print('evaluating network....')
pred_index = model.predict(test_x, batch_size=batch_size)
pred_index = np.argmax(pred_index, axis=1)

print(classification_report(test_y.argmax(axis=1), pred_index, target_names=lb.classes_))

print('saving mask detector model...')
model.save(args['model'],save_format='h5')

N = epochs
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history["acc"], label='train_acc')
plt.plot(np.arange(0,N), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])


