# In order to obtain the faces from the original dataset
# To make it successfully run, you need to create an empty directory named "Processed"

import numpy as np
import json
import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


file = open('./data.json')
file = json.load(file)

classes_to_index = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6}
idx_to_classes = {}
for (k, v) in classes_to_index.items():
    idx_to_classes[v] = k

# Referred to: https://github.com/timesler/facenet-pytorch
mtcnn = MTCNN(image_size=300)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Contain all the faces that are not recognised by Facenet and further recognition is required
missing_image = []

for k, v in file.items():
    root_path = './Processed/'
    image_class = v['Class']
    class_path = root_path + idx_to_classes[image_class]

    if not os.path.isdir(class_path):
        os.mkdir(class_path)

    img = Image.open(v['Image'])
    img_cropped, prob = mtcnn(img, save_path=class_path + '/' + k + '.png', return_prob=True)

    # if there is no face recognised
    if not prob:
        print(k)
        missing_image.append(k)

# These faces cannot recognised by the second recognition and need manual crop
missing_again = []

# set different params to recognise the missing faces
mtcnn = MTCNN(image_size=300, margin=50, thresholds=[0, 0.1, 0.1])
for i in missing_image:
    v = file[i]
    image_path = v['Image']
    root_path = './Processed/'
    image_class = v['Class']
    class_path = root_path + idx_to_classes[image_class]

    img = Image.open(image_path)
    img_cropped, prob = mtcnn(img, save_path=class_path + '/' + i + '.png', return_prob=True)

    # if there is no face recognised again
    if not prob:
        print(i)
        missing_again.append(i)

# obtain the images that require manual crop
print(missing_again)
