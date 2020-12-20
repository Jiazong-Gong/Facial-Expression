# Generate a json file that could help further processing

import json
import os
from os import listdir
from os.path import isfile, join


classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
classes_to_index = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6}
root_path = './Processed/'

images = {}

for i in classes:
    images_path = root_path + i
    imgs = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    for j in imgs:
        name = j.strip('.png')
        img_path = root_path + i + '/' + j
        label = {'Class': classes_to_index[i], 'Image': img_path}
        images[name] = label

json_object = json.dumps(images, indent=4)

# Writing to json file
with open("face_data.json", "w") as outfile:
    outfile.write(json_object)
