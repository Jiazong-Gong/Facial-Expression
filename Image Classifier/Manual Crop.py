# After auto crop, there are 8 faces that cannot be considered as face
# and thus need further processing manually

import json
import os
import cv2
from matplotlib import pyplot as plt


file = open('./data.json')
file = json.load(file)

classes_to_index = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6}
idx_to_classes = {}
for (k, v) in classes_to_index.items():
    idx_to_classes[v] = k

wrongs = ['JennifersBody_001759754_00000001', 'JennifersBody_012514049_00000026',
          'CryingGame_001419640_00000062', 'AlexEmma_004507280_00000060',
          'JennifersBody_012437771_00000033', 'HarryPotter_Deathly_Hallows_1_005528520_00000026',
          'IAmSam_000311160_00000001', 'AlexEmma_010615240_00000022']

i = wrongs[0]  # change the index here for each image above and uncomment corresponding crop below
dim = (300, 300)
image_path = file[i]['Image']
image = cv2.imread(image_path)

# x = 60
# y = 280
# crop = image[x:x+200, y:y+200, :]

# x = 0
# y = 130
# crop = image[x:x+300, y:y+300, :]

# x = 70
# y = 210
# crop = image[x:x+300, y:y+300, :]

# x = 30
# y = 30
# crop = image[x:x+350, y:y+350, :]

# x = 30
# y = 120
# crop = image[x:x+400, y:y+400, :]

# x = 50
# y = 260
# crop = image[x:x+350, y:y+350, :]

# x = 30
# y = 70
# crop = image[x:x+400, y:y+400, :]

# x = 0
# y = 140
# crop = image[x:x+250, y:y+250, :]

# uncomment crop above and these codes to save the cropped faces
# resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)

# compare the original and cropped and decide the area
# plt.subplot(1, 2, 1), plt.imshow(image)
# plt.subplot(1, 2, 2), plt.imshow(resized)
# plt.show()

# cv2.imwrite(i + '.png', resized)

for i in wrongs:
    print(i, idx_to_classes[file[i]['Class']])
