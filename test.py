import pyautogui 
import cv2
import utils
from keras.models import load_model
import time
import numpy as np

def get_state():
    # get image from screen
    img = pyautogui.screenshot()
    # predict box and process
    img, box = utils.predict_and_process(img)
    return img, box

if __name__ == "__main__":
    # laod model
    model = load_model('my_model.h5')

    step = 0
    while True:
        state, box = get_state()
        box = box.reshape(1,box.shape[0]).astype(np.float32)
        action = model.predict(box)
        utils.move(np.argmax(action,axis=1)[0])

        if True:
            cv2.imshow("screenshot", state)
            print("action value: ", np.argmax(action,axis=1)[0], "step: ", step)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break

        step += 1