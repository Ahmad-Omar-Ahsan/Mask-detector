from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np 
import argparse
import cv2 
import os
import imutils 
import time

def detect_and_predict_mask(frame, face_net, mask_net):
    h,w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args['confidence']:
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            x_start, y_start, x_end, y_end = box.astype('int')

            x_start, y_start = (max(0, x_start), max(0, y_start))
            x_end, y_end = (min(w - 1, x_end), min(h - 1, y_end))

            face = frame[y_start:y_end, x_start:x_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((x_start, y_start, x_end, y_end))
        
    if len(faces) > 0:
        faces = np.array(faces, dtype='float32')
        preds = mask_net.predict(faces, batch_size=32)
        
    return locs, preds



ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', type=str, default='face_detector', help='path to face detector model directory')
ap.add_argument('-m', '--model', type=str, default='mask_detector.model', help='path to trained face mask detector model')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='minimum probabiltiy to filter weak detections')
args= vars(ap.parse_args())


print('----- Loading face detector model -----')
proto_txt_path = os.path.sep.join([ args['face'], 'deploy.prototxt'])
weights_path = os.path.sep.join([args['face'], "res10_300x300_ssd_iter_140000.caffemodel"])
face_net = cv2.dnn.readNet(proto_txt_path, weights_path)

print('----- Loading face mask detector model -----')
mask_net = load_model(args['model'])

print('----- Starting video stream -----')
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    locs, preds =detect_and_predict_mask(frame, face_net, mask_net)


    for box, pred in zip(locs, preds):
        x_start, y_start , x_end, y_end= box
        mask, without_mask = pred

        label = 'Mask' if mask > without_mask else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

        label = '{}: {:.2f}'.format(label, max(mask,without_mask)*100)
        cv2.putText(frame, label, (x_start, y_start -10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)

    cv2.imshow('Frame', frame)
    key =cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop

    