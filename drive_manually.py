import copy
import numpy as np
import pyautogui
import cv2
import time
import os
import gzip
import pickle
import keyboard as kb
import utils

def get_state():
    # get image from screen
    img = pyautogui.screenshot()
    # predict box and process
    img , box = utils.predict_and_process(img)
    return img, box

def get_action():
    # action data
    a = np.zeros(5, dtype = np.uint8)

    a[0] = 1 if kb.is_pressed("up") else 0
    a[1] = 1 if kb.is_pressed("left") else 0
    a[2] = 1 if kb.is_pressed("right") else 0
    a[3] = 1 if kb.is_pressed("w") else 0
    a[4] = 1 if kb.is_pressed("s") else 0

    #return(utils.action_arr2id(a))
    return a 

def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)



if __name__ == "__main__":
    step_speed = 0.1
    good_samples = {
        "state" : [],
        "action" : []
    }
    episode_samples = copy.deepcopy(good_samples)
    step = 0

    while True:
        time.sleep(step_speed)

        state, state_box = get_state()
        action = get_action()

        episode_samples["state"].append(state_box)
        episode_samples["action"].append(action)

        # show video and action values for debuging
        if True:
            cv2.imshow("screenshot", state)
            print("action value: ", action, "step: ", step)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break

        # start recording or pressed again to restart recording
        if kb.is_pressed("j"):
            print("start recording in 5s.......")
            time.sleep(5)
            episode_samples["state"] = []
            episode_samples["action"] = []
            step = 0
            
        # stop recording and save data
        if kb.is_pressed("l"):
            print("save data.......")
            time.sleep(5)
            good_samples["state"].append(episode_samples["state"])
            good_samples["action"].append(episode_samples["action"])
            store_data(good_samples, "./data")
            print("save done, there are %d good samples" % len(good_samples["state"]))
            

        step += 1



