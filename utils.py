import numpy as np
from darkflow.net.build import TFNet
from skimage.color import rgb2gray
import cv2
from skimage.draw import polygon_perimeter
from keras.utils import np_utils
import pyautogui


# load the pretrained model and weight
options = {"pbLoad": "built_graph/yolov2-tiny-new.pb", "metaLoad": "built_graph/yolov2-tiny-new.meta", "threshold": 0.1, "gpu": 0.8}
#options = {"pbLoad": "built_graph/tiny-yolo-voc-new.pb", "metaLoad": "built_graph/tiny-yolo-voc-new.meta", "threshold": 0.1, "gpu": 0.8}
tfnet = TFNet(options)

# action define by user
actions = np.array([
    [0, 0, 0, 0, 0],  # STAY
    [1, 0, 0, 0, 0],  # FORWARD
    [0, 0, 1, 0, 0],  # RIGHT
    [0, 1, 0, 0, 0],  # LEFT
    [0, 0, 0, 1, 0],  # UPWARD
    [0, 0, 0, 0, 1],  # DOWNWARD
], dtype=np.uint8)

n_actions = len(actions)


def check_invalid_actions(x, y):
    """ Check if there is any forbidden actions in the expert database """
    inval_list = []

    for idx in range(0, len(y)):
        for action in actions:
            
            if np.array_equal(y[idx], np.array(action)):
                break
            if np.array_equal( np.array(action), np.array(actions[n_actions-1])):
                inval_list.append(idx)
                #print(y[idx])

    x = np.delete(x, inval_list, axis=0)
    y = np.delete(y, inval_list, axis=0)

    return x, y

def move(ids):
    if ids == 0: pass
    if ids == 1: pyautogui.press('up')
    if ids == 2: pyautogui.press('right')
    if ids == 3: pyautogui.press('left')
    if ids == 4: pyautogui.press('w')
    if ids == 5: pyautogui.press('s')

    return 
    

def action_arr2id2agent(arr):
    ids = []
    for a in arr:
        id = np.where(np.all(actions==a, axis=1))
        ids.append(id[0][0])
    
    y = np_utils.to_categorical(ids)
    return y
    
def predict_and_process(image):
    frame = np.array(image)
    result = tfnet.return_predict(frame)
    box_arr = np.zeros(4)
    if len(result) > 0:
        # confidence list
        conf_list = []
        for box in result:
            conf_list.append(box["confidence"])

        conf_arr = np.array(conf_list)
        max_id = np.argmax(conf_arr)
        x1, y1 = result[max_id]["topleft"]["x"], result[max_id]["topleft"]["y"]
        x2, y2 = result[max_id]["bottomright"]["x"], result[max_id]["bottomright"]["y"]
        r = np.array([y1, y1, y2, y2])
        c = np.array([x1, x2, x2, x1])
        rr, cc = polygon_perimeter(r, c)
        frame[:, :] = [0, 0, 0]
        frame[rr, cc] = [255, 255, 255]

        # mid_x, mid_y, w, h
        box_arr = np.array([(((x2-x1)/2)+x1),(((y2-y1)/2)+y1),(x2-x1),(y2-y1)],dtype=np.float32)
        

    else:
        frame[:, :] = [0, 0, 0]

    
    
    # turn shape to (1080,1920,)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, box_arr

def vstack(arr):
    stack = np.array(arr[0][:],dtype=np.float32)
    
    for i in range(1, len(arr)):
        stack = np.vstack((stack, np.array(arr[i][:])))
    return stack
