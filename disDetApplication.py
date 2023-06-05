import os
import numpy as np
import cv2
import keras

# Disease array
printed_diseases = ['citrus_blackspot', 'citrus_canker', 'citrus_fresh', 'guava_canker', 'guava_dot', 'guava_healthy', 'guava_mummification', 'guava_rust', 'mango_anthracnose', 'mango_cutting_weevil', 'mango_die_back', 'mango_gall_midge', 'mango_healthy', 'mango_powdery_mildew', 'mango_sooty_mould']

# Image parameters
klass_dir = 'testPDD'
img_size = 64
num_classes = len(printed_diseases)

# Loading model
model = keras.models.load_model("plantDiseaseModel.h5")

# Initializing arrays
images_to_be_classified = os.listdir(klass_dir)
resized_images = []

# Image preprocessing
for image_name in images_to_be_classified:
    # Construct the full path to the image
    image_path = os.path.join(klass_dir, image_name)
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Resize the image
    resized_image = cv2.resize(image, (img_size, img_size))
    # Append the resized image to the list
    resized_images.append(resized_image)
resized_images = np.array(resized_images)
resized_images = resized_images / 255.0

# Making predictions
predictions = model.predict(resized_images)

# Displaying results
for i in range(len(images_to_be_classified)):
    image_name = images_to_be_classified[i]
    prediction = predictions[i]
    pred_label = ""
    rounded_prediction = np.round(prediction)
    for i in range(len(rounded_prediction)):
        if rounded_prediction[i] == 1:
            diseaseName = printed_diseases[i]
    print("Image:", image_name, "Predicted disease:", diseaseName)