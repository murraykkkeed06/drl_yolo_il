import cv2
import numpy as np
import pyautogui
from darkflow.net.build import TFNet
from skimage.draw import polygon_perimeter
from time import sleep
import math

# display screen resolution, get it from your OS settings
SCREEN_SIZE = (1920, 1080)
# define the codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")
# create the video write object
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (SCREEN_SIZE))
# define the net
options = {"pbLoad": "built_graph/tiny-yolo-voc-new.pb", "metaLoad": "built_graph/tiny-yolo-voc-new.meta", "threshold": 0.1, "gpu": 0.8}
tfnet = TFNet(options)
while True:
    #sleep(1)
    # make a screenshot
    img = pyautogui.screenshot()
    # convert these pixels to a proper numpy array to work with OpenCV
    frame = np.array(img)
    # predict
    result = tfnet.return_predict(frame)
    # draw box from result
    

    if len(result) > 0:
        

        #dist_list = []
        conf_list = []
        # get the biggist box
        for box in result:
            #x1, y1 = box["topleft"]["x"], box["topleft"]["y"]
            #x2, y2 = box["bottomright"]["x"], box["bottomright"]["y"]
            #dist_list.append(math.sqrt((x2-x1)**2+(y2-y1)**2))
            conf_list.append(box["confidence"])
        #dist_arr = np.array(dist_list)
        conf_arr = np.array(conf_list)
       
        max_id = np.argmax(conf_arr)

        print(result[max_id])

        x1, y1 = result[max_id]["topleft"]["x"], result[max_id]["topleft"]["y"]
        x2, y2 = result[max_id]["bottomright"]["x"], result[max_id]["bottomright"]["y"]
        r = np.array([y1, y1, y2, y2])
        c = np.array([x1, x2, x2, x1])
        rr, cc = polygon_perimeter(r, c)
        frame[:, :] = [0, 0, 0]
        frame[rr, cc] = [255, 255, 255]
    
    else:
        frame[:, :] = [0, 0, 0]
        
    #print("checkpoint")
    # convert colors from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #print(frame.shape)
    #(1080, 1920, 3)
    # write the frame
    #out.write(frame)
    # show the frame
    cv2.imshow("screenshot", frame)
    # if the user clicks q, it exits
    if cv2.waitKey(1) == ord("q"):
        break

# make sure everything is closed when exited
cv2.destroyAllWindows()
out.release()