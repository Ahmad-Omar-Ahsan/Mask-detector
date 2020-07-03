from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np 
import argparse
import cv2 
import os

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
ap.add_argument('-f', '--face', type=str, default='face_detector', help='path to face detector model directory')
ap.add_argument('-m', '--model', type=str, default='mask_detector.model', help='path to trained face mask detector model')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='minimum probabiltiy to filter weak detections')
args= vars(ap.parse_args())

print('----- Loading face detector model -----')
proto_txt_path = os.path.sep.join([ args['face'], 'deploy.prototxt'])
weights_path = os.path.sep.join([args['face'], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(proto_txt_path, weights_path)

print('----- Loading face mask detector model -----')
model = load_model(args['model'])


image = cv2.imread(args['image'])
orig = image.copy()
h, w = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

print('----- Computing face detections -----')
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > args['confidence']:
        box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
        x_start, y_start, x_end, y_end = box.astype('int')

        x_start, y_start = (max(0, x_start), max(0, y_start))
        x_end, y_end = (min(w - 1, x_end), min(h - 1, y_end))

        face = image[y_start:y_end, x_start:x_end]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        mask, without_mask = model.predict(face)[0]

        label = 'Mask' if mask > without_mask else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

        label = '{}: {:.2f}'.format(label, max(mask,without_mask)*100)

        cv2.putText(image, label, (x_start, y_start -10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color, 2)

cv2.imshow('Output', image)
cv2.waitKey(0)
