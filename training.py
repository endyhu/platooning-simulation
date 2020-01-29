import cv2
import time
import h5py
import random
import numpy as np
import pyglet as pg
import tensorflow as tf

from collections import deque
from pyglet.window import key
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

DISPLAY = pg.canvas.get_display()
SCREEN = DISPLAY.get_default_screen()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

TITLE = "Platooning Simulator"

background_img = pg.image.load("./assets/map1.png")
car_img = pg.image.load("./assets/car1.png")
sensor_img = pg.image.load("./assets/sensor.png")

background_data = cv2.imread("./assets/map1.png")

def centerImage(image):
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2

def centerCarImage(image):
    image.anchor_x = image.width // 3
    image.anchor_y = image.height // 2

centerCarImage(car_img)

class Object:
    def __init__(self, pos_x, pos_y, image=None):
        if image is not None:
            self.sprite = pg.sprite.Sprite(image, pos_x, pos_y)

    def draw(self):
        self.sprite.draw()

    def update(self, dt):
        pass

class CarObject(Object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_x = self.sprite.x
        self.pos_y = self.sprite.y

        self.acceleration = 0.0
        self.steering = 0
        self.velocity = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.max_acceleration = 20.0
        self.max_velocity = 65.0
        self.max_angular_velocity = 10

    def reset(self):
        self.sprite.x = self.pos_x
        self.sprite.y = self.pos_y
        self.sprite.rotation = 0.0
        self.acceleration = 0.0
        self.velocity = 0.0

    def step(self, action):
        self.acceleration = 0.0
        self.steering = 0
        if action == 1:
            self.acceleration = self.max_acceleration
        elif action == 2:
            self.steering = -1
        elif action == 3:
            self.acceleration = -self.max_acceleration
        elif action == 4:
            self.steering = 1
        elif action == 5:
            self.acceleration = self.max_acceleration
            self.steering = -1
        elif action == 6:
            self.acceleration = self.max_acceleration
            self.steering = 1

    def handleKeys(self):
        self.acceleration = 0.0
        self.steering = 0
        if keys[key.W]:
            self.acceleration = self.max_acceleration
        if keys[key.A]:
            self.steering = -1
        if keys[key.S]:
            self.acceleration = -self.max_acceleration
        if keys[key.D]:
            self.steering = 1

    def update(self, dt):
        self.sprite.rotation = self.sprite.rotation + ((self.steering * 90) * dt)

        self.velocity = self.velocity + self.acceleration * dt
        if self.acceleration == 0.0:
            self.velocity = self.velocity - self.max_acceleration * dt

        if self.velocity > self.max_velocity:
            self.velocity = self.max_velocity
        elif self.velocity < 0.0:
            self.velocity = 0.0

        self.velocity_x = self.velocity * np.cos(np.deg2rad(self.sprite.rotation))
        self.velocity_y = self.velocity * -np.sin(np.deg2rad(self.sprite.rotation))

        self.sprite.x = self.sprite.x + self.velocity_x * dt
        self.sprite.y = self.sprite.y + self.velocity_y * dt

class LineDetectors:
    def __init__(self, car, show=False):
        self.car = car
        self.sprite_batch = pg.graphics.Batch()
        self.detector_sprites = []

        if show:
            for _ in range(2):
                self.detector_sprites.append(pg.sprite.Sprite(sensor_img, self.car.sprite.x, self.car.sprite.y, batch=self.sprite_batch))

    def getData(self, is_bool=False):
        output_data = []
        output_data_bool = []

        for sprite in self.detector_sprites:
            data = background_data[600 - int(sprite.y), int(sprite.x)]
            data_bool = 0
            if np.sum(data) > 30:
                data_bool = 1

            output_data.append(data)
            output_data_bool.append(data_bool)

        if is_bool:
            return np.array(output_data_bool)
        return np.array(output_data)

    def draw(self):
        self.sprite_batch.draw()

    def update(self, dt):
        pos_x = self.car.sprite.x + (self.car.sprite.width / 3 * 2) * np.cos(np.deg2rad(self.car.sprite.rotation))
        pos_y = self.car.sprite.y + (self.car.sprite.width / 3 * 2) * -np.sin(np.deg2rad(self.car.sprite.rotation))

        for i, sprite in enumerate(self.detector_sprites):
            offset_x = (1 - i * 2) * (self.car.sprite.height / 2) * np.sin(np.deg2rad(self.car.sprite.rotation))
            offset_y = (1 - i * 2) * (self.car.sprite.height / 2) * np.cos(np.deg2rad(self.car.sprite.rotation))

            sprite.x = pos_x + offset_x
            sprite.y = pos_y + offset_y

OBSERVATION_SPACE_N = 5
ACTION_SPACE_N = 7

class Estimator:
    def __init__(self):
        self.model = Sequential()
        
        self.model.add(Dense(32, input_shape=(OBSERVATION_SPACE_N,)))
        self.model.add(Activation("relu"))
        self.model.add(Dense(64))
        self.model.add(Activation("relu"))
        
        self.model.add(Dense(ACTION_SPACE_N))
        
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=self.optimizer, 
                           loss="logcosh")
        
        self.model.summary()
        
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
    def preprocess(self, state):
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
        
    def predictTarget(self, state):
        state = self.preprocess(state)
        prediction = self.target_model.predict(state)
        
        return prediction
        
    def updateTarget(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def save(self, filename):
        self.model.save(f"./models/{filename}.h5")
        
    def load(self, filename):
        self.model.load_weights(f"./models/{filename}.h5")
        self.updateTarget()

estimator = Estimator()
# estimator.load("_705.76_00057137_06_1452")

MAX_STEPS = 1000000
MAX_EPISODE_STEPS = 9000

DISCOUNT = 0.99
BATCH_SIZE = 32

EPSILON_INIT = 1.0
EPSILON_MIN = 0.1
EPSILON_END = 10000

REPLAY_MEMORY_SIZE = 100000
REPLAY_START_SIZE = 10000

UPDATE_FREQ = 4
TARGET_NETWORK_UPDATE_FREQ = 10000

def EpsilonGreedyPolicy(state, epsilon):
    A = np.ones(ACTION_SPACE_N) * (epsilon / ACTION_SPACE_N)
    best_action = np.argmax(estimator.predict(state))
    A[best_action] = A[best_action] + (1 - epsilon)

    return A

class Window(pg.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_rate = 1/60.0

        location_x = (SCREEN.width // 2) - (WINDOW_WIDTH // 2)
        location_y = (SCREEN.height // 2) - (WINDOW_HEIGHT // 2)
        self.set_location(location_x, location_y)

        self.background = pg.sprite.Sprite(background_img, x=0, y=0)
        
        self.car = CarObject(WINDOW_WIDTH/2, WINDOW_HEIGHT/2, car_img)
        self.line_detectors = LineDetectors(self.car, True)

    def getState(self):
        acceleration = self.car.acceleration / self.car.max_acceleration
        velocity = self.car.velocity / self.car.max_velocity
        line0, line1 = self.line_detectors.getData(True)
        distance = 1.0

        return np.array([acceleration, velocity, line0, line1, distance])

    def reset(self):
        self.car.reset()

        return self.getState()

    def step(self, action):
        self.car.step(action)
        self.update(1/30)

        state = self.getState()
        reward = -0.8
        done = False

        if state[0] >= 0.5:
            reward = 0.2
        if state[1] >= 0.5:
            reward = state[1]
        if state[2] or state[3]:
            reward = reward - 0.2
        if state[2] and state[3]:
            reward = -1.0
            done = True

        return state, reward, done, None

    def QLearning(self):
        num_steps = 0
        episode_rewards = []

        epsilon = EPSILON_INIT
        epsilon_gradient = (EPSILON_INIT - EPSILON_MIN) / EPSILON_END
        
        replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        state = self.reset()
        for _ in range(REPLAY_START_SIZE):
            action_prob = EpsilonGreedyPolicy(state, epsilon)
            action = np.random.choice([i for i in range(ACTION_SPACE_N)], p=action_prob)

            next_state, reward, done, _ = self.step(action)

            replay_memory.append((state, action, reward, next_state, done))
            print(f"Replay Memory Start Size: ({len(replay_memory)}/{REPLAY_START_SIZE})")

            if done:
                state = self.reset()

            state = next_state

        while num_steps < MAX_STEPS:
            state = self.reset()
            episode_reward = 0

            for _ in range(MAX_EPISODE_STEPS):
                action_prob = EpsilonGreedyPolicy(state, epsilon)
                action = np.random.choice([i for i in range(ACTION_SPACE_N)], p=action_prob)

                next_state, reward, done, _ = self.step(action)

                num_steps = num_steps + 1
                episode_reward = episode_reward + reward
                replay_memory.append((state, action, reward, next_state, done))

                if epsilon > EPSILON_MIN:
                    epsilon = epsilon - epsilon_gradient

                if num_steps % UPDATE_FREQ == 0:
                    replay_batch = random.sample(replay_memory, BATCH_SIZE)

                    for ss, aa, rr, ns, terminal in replay_batch:
                        td_target = rr

                        if not terminal:
                            best_next_action_value = np.argmax(estimator.predictTarget(ns))
                            td_target = rr + DISCOUNT * best_next_action_value

                        estimator.update(ss, aa, td_target)

                if num_steps % TARGET_NETWORK_UPDATE_FREQ == 0:
                    estimator.updateTarget()

                if done:
                    break

                state = next_state

            if len(episode_rewards) == 0 or episode_reward >= max(episode_rewards):
                local_time = time.localtime()
                estimator.save(f"{episode_reward:.2f}_{num_steps:>08}_{local_time.tm_mday:>02}_{local_time.tm_hour:>02}{local_time.tm_min:>02}")

            episode_rewards.append(episode_reward)
            print(f"[{len(episode_rewards)}] ({num_steps}/{MAX_STEPS}) Episode Reward: {episode_rewards[-1]:.5f} Epsilon: {epsilon}")

        local_time = time.localtime()
        filename = f"{episode_reward:.2f}_{num_steps:>08}_{local_time.tm_mday:>02}_{local_time.tm_hour:>02}{local_time.tm_min:>02}"
        estimator.save(filename)
        print("Model Saved")
            
        np.savetxt(f"./models/{filename}.csv", episode_rewards)
        print("Episode Rewards Saved")

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        action = np.argmax(estimator.predict(self.getState()))
        print(self.step(action))

    def on_key_press(self, symbol, modifier):
        if symbol == key.R:
            self.reset()
        if symbol == key.ENTER:
            self.QLearning()

    def on_draw(self):
        self.clear()
        self.background.draw()
        self.car.draw()
        self.line_detectors.draw()

    def update(self, dt):
        # self.car.handleKeys()
        action = np.argmax(estimator.predict(self.getState()))
        self.car.step(action)
        self.car.update(dt)
        self.line_detectors.update(dt)

if __name__ == "__main__":
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    # pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
