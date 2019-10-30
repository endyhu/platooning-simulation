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

background_img = pg.image.load("./assets/map2.png")
car_img = pg.image.load("./assets/car.png")
sensor_img = pg.image.load("./assets/sensor.png")

background_data = cv2.imread("./assets/map2.png")

def centerImage(image):
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2

def centerCarImage(image):
    image.anchor_x = image.width // 3
    image.anchor_y = image.height // 2

centerCarImage(car_img)

class CarObject:
    def __init__(self, pos_x, pos_y, image=None):
        if image is not None:
            self.sprite = pg.sprite.Sprite(image, pos_x, pos_y)

        self.pos_x = pos_x
        self.pos_y = pos_y

        self.acceleration = 0.0
        self.steering = 0
        self.velocity = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.max_acceleration = 20.0
        self.max_velocity = 65.0
        self.max_angular_velocity = 10

        self.sensor_line = SensorLine(self, True)
        self.sensor_distance = SensorDistance(self, True)

    def getState(self, cars):
        acceleration = self.acceleration / self.max_acceleration
        velocity = self.velocity / self.max_velocity
        line0, line1 = self.sensor_line.getData(True)
        distance = self.sensor_distance.getData(cars) / self.sensor_distance.max_distance

        return np.array([acceleration, velocity, line0, line1, distance])

    def reset(self):
        self.sprite.x = self.pos_x
        self.sprite.y = self.pos_y
        self.sprite.rotation = 0.0
        self.acceleration = 0.0
        self.velocity = 0.0

    def step(self, action, distance):
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

        brake_distance = (self.velocity**2) / (2 * self.max_acceleration)
        if (distance - 12.0) <= brake_distance:
            self.acceleration = -self.max_acceleration

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

    def draw(self):
        self.sprite.draw()
        self.sensor_line.draw()
        self.sensor_distance.draw()

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

        self.sensor_line.update(dt)
        self.sensor_distance.update(dt)

class SensorLine:
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

class SensorDistance:
    def __init__(self, car, show=False):
        self.car = car
        self.sprite = pg.sprite.Sprite(sensor_img, 0, 0)
        self.max_angle = 360.0
        self.max_distance = 100.0

    def getData(self, cars):
        angle = self.max_angle
        distance = self.max_distance

        b_x = self.sprite.x + np.cos(np.deg2rad(self.car.sprite.rotation))
        b_y = self.sprite.y + -np.sin(np.deg2rad(self.car.sprite.rotation))
        b = np.array([b_x, b_y])

        for car in cars:
            if car != self.car:
                a = np.array([self.sprite.x, self.sprite.y])
                c = np.array([car.sprite.x, car.sprite.y])

                d = b - a
                e = c - a

                dm = (d[0]**2 + d[1]**2)**0.5
                em = (e[0]**2 + e[1]**2)**0.5

                car_angle = np.rad2deg(np.arccos(np.dot(d, e)/(dm * em)))
                car_distance = abs(em - car.sprite.width / 3)

                if car_angle <= self.max_angle and car_distance <= distance:
                    angle = car_angle
                    distance = car_distance

        return distance

    def draw(self):
        self.sprite.draw()

    def update(self, dt):
        self.sprite.x = self.car.sprite.x + (self.car.sprite.width / 3 * 2) * np.cos(np.deg2rad(self.car.sprite.rotation))
        self.sprite.y = self.car.sprite.y + (self.car.sprite.width / 3 * 2) * -np.sin(np.deg2rad(self.car.sprite.rotation))

OBSERVATION_SPACE_N = 5
ACTION_SPACE_N = 7

class Estimator:
    def __init__(self):
        self.model = Sequential()
        
        self.model.add(Dense(32, input_shape=(OBSERVATION_SPACE_N,)))
        self.model.add(Activation("relu"))
        # self.model.add(Dense(32))
        # self.model.add(Activation("relu"))
        self.model.add(Dense(64))
        self.model.add(Activation("relu"))
        
        self.model.add(Dense(ACTION_SPACE_N))
        
        self.optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=self.optimizer, 
                           loss="logcosh")
        
        self.model.summary()
        
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        
    def preprocess(self, state):
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
estimator.load("_705.76_00057137_06_1452")

MAX_STEPS = 1000000
MAX_EPISODE_STEPS = 1000

DISCOUNT = 0.99
BATCH_SIZE = 32

EPSILON_INIT = 1.0
EPSILON_MIN = 0.1
EPSILON_END = 100000

REPLAY_MEMORY_SIZE = 100000
REPLAY_START_SIZE = 50000

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
        
        self.car0 = CarObject(WINDOW_WIDTH/2, WINDOW_HEIGHT/2, car_img)

        self.obs_idx = np.random.randint(0, 9)
        self.obs_counter = 0
        self.obs_data = [
            [594.89463, 185.91862,   91.06800],
            [688.71395,  41.85971,    0.32730],
            [735.40366, 413.63733,  -90.47457],
            [651.71919, 526.01366, -175.59751],
            [314.79445, 496.40442, -185.99217],
            [109.75403, 288.53455, -274.26786],
            [111.37117,  37.04259, -323.63017],
            [246.56537, 281.18256, -409.39490],
            [469.66971, 297.85476, -359.16123]]

        self.car_obs = CarObject(self.obs_data[self.obs_idx][0], self.obs_data[self.obs_idx][1], car_img)
        self.car_obs.sprite.rotation = self.obs_data[self.obs_idx][2]

        self.cars = [self.car0, self.car_obs]

    def reset(self):
        for car in self.cars:
            car.reset()

        self.obs_idx = np.random.randint(0, 9)
        self.obs_counter = 0
        self.car_obs.sprite.x = self.obs_data[self.obs_idx][0]
        self.car_obs.sprite.y = self.obs_data[self.obs_idx][1]
        self.car_obs.sprite.rotation = self.obs_data[self.obs_idx][2]

        return self.car0.getState(self.cars)

    def step(self, action):
        state = self.car0.getState(self.cars)

        self.car0.step(action, (state[4] * 100))
        self.update(1/30)

        state = self.car0.getState(self.cars)
        reward = 0.0
        done = False
        
        if state[1] > 0.2:
            reward = state[1]
        if state[4] < (14.0 / self.car0.sensor_distance.max_distance):
            reward = 1.0
        if state[2] and state[3]:
            reward = -1.0
            done = True
        if state[4] < (8.0 / self.car0.sensor_distance.max_distance):
            reward = -1.0
            done = True

        return state, reward, done, None

    def continuousStep(self):
        state = self.car0.getState(self.cars)
        action = np.argmax(estimator.predict(state))
        state = self.car0.getState(self.cars)
        self.car0.step(action, (state[4] * 100))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        action = np.argmax(estimator.predict(self.car0.getState(self.cars)))
        print(self.step(action))

    def on_key_press(self, symbol, modifier):
        if symbol == key.R:
            self.reset()
        if symbol == key.SPACE:
            print(f"{self.car0.sprite.x:.05f}, {self.car0.sprite.y:.05f}, {self.car0.sprite.rotation:.05f}")

    def on_draw(self):
        self.clear()
        self.background.draw()

        for car in self.cars:
            car.draw()

    def updateObs(self):
        if self.car0.sensor_distance.getData(self.cars) < 50.0:
            self.obs_counter = self.obs_counter + 1
            if self.obs_counter > 150:
                self.obs_idx = np.random.randint(0, 9)
                self.car_obs.sprite.x = self.obs_data[self.obs_idx][0]
                self.car_obs.sprite.y = self.obs_data[self.obs_idx][1]
                self.car_obs.sprite.rotation = self.obs_data[self.obs_idx][2]
                self.obs_counter = 0

    def update(self, dt):
        # self.car0.handleKeys()
        self.continuousStep()

        for car in self.cars:
            car.update(dt)

        self.updateObs()
        # print(self.car0.sensor_distance.getData(self.cars))

if __name__ == "__main__":
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
