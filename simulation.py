import cv2
import numpy as np
import pyglet as pg

from pyglet.window import key

DISPLAY = pg.canvas.get_display()
SCREEN = DISPLAY.get_default_screen()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

KEYS = key.KeyStateHandler()

MAP_IMG = "./assets/map2.png"

background_img = pg.image.load(MAP_IMG)
car1_img = pg.image.load("./assets/car1.png")
car2_img = pg.image.load("./assets/car2.png")
sensor_img = pg.image.load("./assets/sensor.png")

background_data = cv2.imread(MAP_IMG)

def adjustImageAnchor(image, x, y):
    image.anchor_x = image.width // x
    image.anchor_y = image.height // y

adjustImageAnchor(car1_img, 3, 2)
adjustImageAnchor(car2_img, 3, 2)

class SensorLine:
    def __init__(self, car, show=False):
        self.car = car
        self.show = show
        self.detector_sprites = []
        self.sprite_batch = pg.graphics.Batch()

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
        if self.show:
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
    def __init__(self, car, max_angle=90.0, max_distance=450.0, show=False):
        self.car = car
        self.sprite = pg.sprite.Sprite(sensor_img, 0, 0)
        self.max_angle = max_angle
        self.max_distance = max_distance
        self.show = show

    def getData(self, cars):
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
                    distance = car_distance

        return distance

    def draw(self):
        if self.show:
            self.sprite.draw()

    def update(self, dt):
        self.sprite.x = self.car.sprite.x + (self.car.sprite.width / 3 * 2) * np.cos(np.deg2rad(self.car.sprite.rotation))
        self.sprite.y = self.car.sprite.y + (self.car.sprite.width / 3 * 2) * -np.sin(np.deg2rad(self.car.sprite.rotation))

class CarObject:
    def __init__(self, pos_x, pos_y, optimal_distance=33.0, image=car1_img):
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

        self.optimal_distance = optimal_distance

        self.sensor_line = SensorLine(self, show=True)
        self.sensor_distance = SensorDistance(self, show=True)

        self.estimator = None

    def addEstimator(self, estimator):
        self.estimator = estimator

    def predict(self, state):
        action = np.argmax(self.estimator.predict(state))

        return action

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

        distance = distance * self.sensor_distance.max_distance

        brake_distance = (self.velocity**2) / (2 * self.max_acceleration)
        print(distance, brake_distance)
        if (distance - self.optimal_distance) <= brake_distance:
            self.acceleration = -self.max_acceleration

    def handleKeys(self):
        self.acceleration = 0.0
        self.steering = 0
        if KEYS[key.W]:
            self.acceleration = self.max_acceleration
        if KEYS[key.A]:
            self.steering = -1
        if KEYS[key.S]:
            self.acceleration = -self.max_acceleration
        if KEYS[key.D]:
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

class ObstacleObject:
    def __init__(self, distance, max_counter, image=car2_img):
        self.idx = 0
        self.counter = 0
        self.max_counter = max_counter
        self.distance = distance
        self.data = [
            [594.89463, 185.91862,   91.06800],
            [688.71395,  41.85971,    0.32730],
            [735.40366, 413.63733,  -90.47457],
            [651.71919, 526.01366, -175.59751],
            [314.79445, 496.40442, -185.99217],
            [109.75403, 288.53455, -274.26786],
            [111.37117,  37.04259, -323.63017],
            [246.56537, 281.18256, -409.39490],
            [469.66971, 297.85476, -359.16123]]

        self.car = CarObject(0, 0, 33.0, image)
        self.car.sensor_distance = SensorDistance(self.car, 180.0, 100.0, True)

    def reset(self):
        self.idx = np.random.randint(0, len(self.data))
        self.counter = 0

    def draw(self):
        self.car.draw()
    
    def update(self, cars, dt):
        if self.car.sensor_distance.getData(cars) <= (self.distance + 22.0):
            self.counter = self.counter + 1
            if self.counter >= self.max_counter:
                self.idx = np.random.randint(0, len(self.data))
                self.counter = 0

        self.car.sprite.x = self.data[self.idx][0]
        self.car.sprite.y = self.data[self.idx][1]
        self.car.sprite.rotation = self.data[self.idx][2]
        self.car.update(dt)

class Window(pg.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_rate = 1/60.0

        location_x = (SCREEN.width // 2) - (WINDOW_WIDTH // 2)
        location_y = (SCREEN.height // 2) - (WINDOW_HEIGHT // 2)
        self.set_location(location_x, location_y)

        self.background = pg.sprite.Sprite(background_img, x=0, y=0)
        
        self.obstacle = ObstacleObject(66.0, 300)
        self.cars = []

    def createCar(self, pos_x, pos_y, optimal_distance=33.0, image=car1_img):
        new_car = CarObject(pos_x, pos_y, optimal_distance, image)
        self.cars.append(new_car)

        return new_car

    def reset(self):
        self.obstacle.reset()
        for car in self.cars:
            car.reset()

    def on_key_press(self, symbol, modifier):
        if symbol == key.R:
            self.reset()

    def on_draw(self):
        self.clear()
        self.background.draw()

        for car in self.cars:
            car.draw()

        self.obstacle.draw()

    def update(self, dt):
        # if len(self.cars):
        #     self.cars[0].handleKeys()

        for car in self.cars:
            if car.estimator:
                all_cars = self.cars + [self.obstacle.car]
                state = car.getState(all_cars)
                action = car.predict(state)
                car.step(action, state[4])
            car.update(dt)

        self.obstacle.update(self.cars, dt)
