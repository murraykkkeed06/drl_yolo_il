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
    img = utils.predict_and_process(img)
    return img

if __name__ == "__main__":
    # laod model
    model = load_model('my_model.h5')

    steo = 0
    while True:
        state = get_state()
        predict_result = model.predict(state)
        utils.move(np.argmax(predict_result,axis=1)[0])

        if True:
            cv2.imshow("screenshot", state)
            print("action value: ", action, "step: ", step)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break

        step += 1