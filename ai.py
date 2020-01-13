import os
import h5py
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

OBSERVATION_SPACE_N = 5
ACTION_SPACE_N = 7

# OBSERVATION_SPACE_N
# [acceleration, velocity, line0, line1, distance]

# ACTION_SPACE_N
# [DO_NOTHING, ACCELERATE, LEFT, BRAKE, RIGHT, ACCELERATE+LEFT, ACCELERATE+RIGHT]

class Estimator:
    def __init__(self):
        self.model = Sequential()
        
        self.model.add(Dense(32, input_shape=(OBSERVATION_SPACE_N,)))
        self.model.add(Activation("relu"))
        self.model.add(Dense(64))
        self.model.add(Activation("relu"))
        
        self.model.add(Dense(ACTION_SPACE_N))
        
        self.optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=self.optimizer, 
                           loss="logcosh")
        
        self.model.summary()
        
    def preprocess(self, pre_state):
        state = np.copy(pre_state)
        state[4] = 1.0

        return state.reshape(-1, OBSERVATION_SPACE_N)
    
    def predict(self, state):
        state = self.preprocess(state)
        prediction = self.model.predict(state)
        
        return prediction
    
    def update(self, s, a, y):
        state = self.preprocess(s)
        
        td_target = self.predict(s)
        td_target[0][a] = y
        
        self.model.train_on_batch(state, td_target)
        
    def save(self, filename):
        self.model.save(f"./models/{filename}.h5")
        
    def load(self, filename):
        self.model.load_weights(f"./models/{filename}.h5")
