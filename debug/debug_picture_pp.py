from darkflow.net.build import TFNet
import cv2
from PIL import Image
import numpy as np
from numpy import asarray
from skimage.color import rgb2gray
from skimage.draw import polygon_perimeter
import math

PICTURE_PATH = "images/1.PNG"


def replace_color(old_color, new_color):
    mask = np.all(states_pp == old_color, axis=3)
    states_pp[mask] = new_color

#detect box and print result
options = {"pbLoad": "built_graph/tiny-yolo-voc-new.pb", "metaLoad": "built_graph/tiny-yolo-voc-new.meta", "threshold": 0.1, "gpu": 0.8}
tfnet = TFNet(options)
imgcv = cv2.imread(PICTURE_PATH)
result = tfnet.return_predict(imgcv)
print(result)

#read image and turn to greyscale
image = Image.open(PICTURE_PATH)
data = asarray(image)
data_rgb = data[:,:,:3]
data_pp = rgb2gray(data_rgb/255)
data_pp = np.uint8(data_pp*255)

print(data_pp.shape)

dist_list = []
#get the biggist box
for box in result:
    if box["confidence"] > 0.5:
        x1, y1 = box["topleft"]["x"], box["topleft"]["y"]
        x2, y2 = box["bottomright"]["x"], box["bottomright"]["y"]
        dist_list.append(math.sqrt((x2-x1)**2+(y2-y1)**2))
        
dist_arr = np.array(dist_list)
max_id = np.argmax(dist_arr)

x1, y1 = result[max_id]["topleft"]["x"], result[max_id]["topleft"]["y"]
x2, y2 = result[max_id]["bottomright"]["x"], result[max_id]["bottomright"]["y"]
r = np.array([y1, y1, y2, y2])
c = np.array([x1, x2, x2, x1])
rr, cc = polygon_perimeter(r, c)
data_pp[:, :] = 0
data_pp[rr, cc] = 255

"""data_pp[:y1,:] = 0
data_pp[y2:,:] = 0
data_pp[:,:x1] = 0
data_pp[:,x2:] = 0"""


img = Image.fromarray(data_pp , 'L')
img.show()



"""
img = Image.fromarray(data)
draw = ImageDraw.Draw(img)
draw.polygon([tuple(p) for p in poly], fill=0)
new_data = np.asarray(img)"""