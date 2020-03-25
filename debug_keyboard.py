import keyboard  as kb
import numpy as np




actions = np.array([
    [0, 0, 0, 0, 0],  # STAY
    [1, 0, 0, 0, 0],  # FORWARD
    [0, 0, 1, 0, 0],  # RIGHT
    [0, 1, 0, 0, 0],  # LEFT
    [0, 0, 0, 1, 0],  # UPWARD
    [0, 0, 0, 0, 1],  # DOWNWARD
], dtype=np.uint8)

n_actions = len(actions)

def action_arr2id(arr):
    ids = []
    for action in actions:
        ids.append(np.all(action==arr))
    
    return np.argmax(np.array(ids,dtype=np.uint8))
    

if __name__ == "__main__":

    a = np.zeros(5, dtype = np.uint8)

    while True:  # making a loop

        a[0] = 1 if kb.is_pressed("up") else 0
        a[1] = 1 if kb.is_pressed("left") else 0
        a[2] = 1 if kb.is_pressed("right") else 0
        a[3] = 1 if kb.is_pressed("w") else 0
        a[4] = 1 if kb.is_pressed("s") else 0
       

        print(action_arr2id(a))



       
    