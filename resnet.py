import os
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from scipy.spatial import distance
import shutil
import time

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(folder_path, model):
    features = {}
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        processed_img = load_and_preprocess_image(img_path)
        img_features = model.predict(processed_img)
        features[img_path] = img_features.flatten()
    return features

original_folder = 'imageset\\original'
test_folder = 'imageset\\test'
duplicates_folder = 'imageset\\duplicates'
os.makedirs(duplicates_folder, exist_ok=True)

model = ResNet50(weights='imagenet', include_top=False)

start_time = time.time()
original_features = extract_features(original_folder, model)
test_features = extract_features(test_folder, model)

threshold = 0.9
with open('CNN_result.txt', 'w') as log_file:
    for test_path, test_feat in test_features.items():
        for original_path, original_feat in original_features.items():
            sim = 1 - distance.cosine(test_feat, original_feat)
            if sim > threshold:
                test_filename = os.path.basename(test_path)
                original_filename = os.path.basename(original_path)

                log_file.write(f"Duplicate for: {test_filename} : {original_filename}, Similarity: {sim}\n")
                shutil.move(test_path, os.path.join(duplicates_folder, test_filename))

end_time = time.time()
execution_time = end_time - start_time

with open('CNN_result.txt', 'a') as log_file:
    log_file.write(f"Execution Time: {execution_time} seconds\n")