import cv2
import numpy as np
import pyglet as pg
import ast
import json
import keras as krs

from pyglet.window import key

DISPLAY = pg.canvas.get_display()
SCREEN = DISPLAY.get_default_screen()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

TITLE = "Platooning Simulator"

background_img = pg.image.load("./assets/map.png")
car_img = pg.image.load("./assets/car.png")
sensor_img = pg.image.load("./assets/sensor.png")

background_data = cv2.imread("./assets/map.png")

model = krs.models.load_model('./supervised/models/car_ai.hdf5')


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
        self.acceleration = 0.0
        self.steering = 0
        self.velocity = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.max_acceleration = 20.0
        self.max_velocity = 40.0
        self.max_angular_velocity = 10

    def step(self, action):
        self.acceleration = 0.0
        self.steering = 0
        if (action == [1, 0, 0]).all():
            self.acceleration = self.max_acceleration
            # print('F')
        elif (action == [0, 1, 0]).all():
            self.steering = -1
            print('L')
        elif (action == [0, 0, 0]).all():
            self.acceleration = self.max_acceleration
        elif (action == [0, 0, 1]).all():
            print('R')
            self.steering = 1
        elif (action == [1, 0, 0]).all():
            print('FL')
            self.acceleration = self.max_acceleration
            self.steering = -1
        elif (action == [1, 0, 1]).all():
            # print('FR')
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
                self.detector_sprites.append(pg.sprite.Sprite(sensor_img, 0, 0, batch=self.sprite_batch))

    def getData(self):
        output_data = []

        for sprite in self.detector_sprites:
            data = background_data[600 - int(sprite.y), int(sprite.x)]
            output_data.append(data)

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

class Window(pg.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_rate = 1/60.0

        location_x = (SCREEN.width // 2) - (WINDOW_WIDTH // 2)
        location_y = (SCREEN.height // 2) - (WINDOW_HEIGHT // 2)
        self.set_location(location_x, location_y)

        self.background = pg.sprite.Sprite(background_img, x=0, y=0)
        self.record_sensordata = [np.zeros(shape=(2, 3), dtype='uint8')]
        self.input_keys = []
        self.record_cardata = []
        self.test_record = []

        self.car = CarObject(WINDOW_WIDTH/2, WINDOW_HEIGHT/2, car_img)
        self.line_detectors = LineDetectors(self.car, True)

    def on_draw(self):
        self.clear()
        self.background.draw()
        self.car.draw()
        self.line_detectors.draw()

    def handleKeys(self):
        if keys[key.P]:
            sensor = np.array(self.record_sensordata)
            cardata = np.array(self.record_cardata)

            np.save('./supervised/data/sensor_data.npy', sensor)
            np.save('./supervised/data/car_data.npy', cardata)

            with open('./supervised/data/input_keys.json', 'w') as f:
                json.dump(self.input_keys, f)
        if keys[key.L]:
            test_rec = np.array(self.test_record)
            np.save('./supervised/data/test_record.npy', test_rec)

    def sensor_preprocess(self, data):
        if np.array_equal(data, [[0, 0, 0], [0, 0, 0]]):
            res = [1, 0, 0, 0, 0]  # R-R
        elif np.array_equal(data, [[0, 0, 0], [254, 254, 254]]) or np.array_equal(data, [[0, 0, 0], [255, 255, 255]]) or np.array_equal(data, [[0, 0, 0], [254, 255, 254]]):
            res = [0, 1, 0, 0, 0]  # R-L
        elif np.array_equal(data, [[0, 0, 0], [0, 188, 0]]):
            res = [0, 0, 1, 0, 0]  # R-G
        elif np.array_equal(data, [[254, 254, 254], [0, 0, 0]]) or np.array_equal(data, [[255, 255, 255], [0, 0, 0]]) or np.array_equal(data, [[254, 255, 254], [0, 0, 0]]):
            res = [0, 0, 0, 1, 0]  # R-G#L-R
        elif np.array_equal(data, [[0, 188, 0], [0, 0, 0]]):
            res = [0, 0, 0, 0, 1]  # G-R
        else:
            res = [0, 0, 0, 0, 0]

        return res

    def update(self, dt):
        self.handleKeys()
        # self.car.handleKeys()
        self.car.update(dt)
        self.line_detectors.update(dt)

        # self.input_keys.append(ast.literal_eval(str(keys)))
        # self.record_sensordata.append(np.array(self.line_detectors.getData()))
        # self.record_cardata.append(np.array([self.car.steering,
        #                            self.car.velocity,
        #                            self.car.velocity_x,
        #                            self.car.velocity_y]))


        # predict data
        prs_sensor = self.sensor_preprocess(self.line_detectors.getData())
        prs_sensor.extend([self.car.steering, self.car.velocity, self.car.velocity_x, self.car.velocity_y])
        self.car.step(np.round(model.predict(np.array([prs_sensor]))))

        print(np.round(model.predict(np.array([prs_sensor])), 2))
        #print(self.car.velocity_x, self.car.velocity_y, self.car.steering)

        self.test_record.append(np.array(prs_sensor))

        # print(self.line_detectors.getData())




if __name__ == "__main__":
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
