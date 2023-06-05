import os
import numpy as np
import cv2
import keras

# Disease array
printed_diseases = ['citrus_blackspot', 'citrus_canker', 'citrus_fresh', 'guava_canker', 'guava_dot', 'guava_healthy', 'guava_mummification', 'guava_rust', 'mango_anthracnose', 'mango_cutting_weevil', 'mango_die_back', 'mango_gall_midge', 'mango_healthy', 'mango_powdery_mildew', 'mango_sooty_mould']

# Image parameters
img_size = 64
num_classes = len(printed_diseases)

# Loading model
model = keras.models.load_model("plantDiseaseModel.h5")

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    resized_frame = cv2.resize(frame, (img_size, img_size))
    resized_frame = np.expand_dims(resized_frame, axis=0)
    resized_frame = resized_frame / 255.0

    prediction = model.predict(resized_frame)[0]
    pred_label = printed_diseases[np.argmax(prediction)]

    cv2.putText(frame, pred_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()