from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import gzip
import os
import utils


def read_good_samples():
    with gzip.open('./data/data.pkl.gzip','rb') as f:
        data = pickle.load(f)
        print(len(data["state"]))
        # read and stack
        x = utils.vstack(data["state"])
        y = utils.vstack(data["action"])
        # delete the not defined action
        x, y = utils.check_invalid_actions(x, y)
   
    return x, y

def preprocess_data(x, y):
    #x_pp = x.reshape(x.shape[0],1080,1920,1).astype("float32")
    y_pp = utils.action_arr2id2agent(y)

    return x, y_pp 

def net():
    # mlp
    
    model = Sequential()
    model.add(Dense(units=128, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=utils.n_actions, kernel_initializer='normal',activation='softmax'))
    

    

    return model

def plot_action_histogram(actions, title):
    """ Plot the histogram of actions from the expert dataset """
    acts_id = np.argmax(actions,axis=1)
    fig, ax = plt.subplots()
    bins = np.arange(-.5, utils.n_actions + .5)
    ax.hist(acts_id, range=(0,6), bins=bins, rwidth=.9)
    ax.set(title=title, xlim=(-.5, utils.n_actions -.5))
    plt.show()


if __name__ == "__main__":
    x, y = read_good_samples()
    print(x.shape)
    # turn to catgorical style and reshape
    x_pp, y_pp = preprocess_data(x, y)

    print(y_pp.shape)
    if True: print(x_pp.shape, y_pp.shape)
    
    if False: plot_action_histogram(y_pp, 'Action distribution BEFORE balancing')   
 
    if False: x_pp, y_pp = utils.balance_actions(x_pp, y_pp, 0.9)

    if False: plot_action_histogram(y_pp, 'Action distribution AFTER balancing')   
 
    # define net
    model = net()
    # train
    if True:
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.fit(x=x_pp,y=y_pp,validation_split=0.2,epochs=50,batch_size=4,verbose=2)
        model.save('my_model.h5')


    


