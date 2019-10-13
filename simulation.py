import cv2
import numpy as np
import pyglet as pg

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

class DistanceModule:
    def __init__(self, car, show=False):
        self.car = car
        self.sprite = pg.sprite.Sprite(sensor_img, 0, 0)
        self.max_angle = 15.0
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

        return np.array([angle, distance])

    def draw(self):
        self.sprite.draw()

    def update(self, dt):
        self.sprite.x = self.car.sprite.x + (self.car.sprite.width / 3 * 2) * np.cos(np.deg2rad(self.car.sprite.rotation))
        self.sprite.y = self.car.sprite.y + (self.car.sprite.width / 3 * 2) * -np.sin(np.deg2rad(self.car.sprite.rotation))

class Window(pg.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_rate = 1/60.0

        location_x = (SCREEN.width // 2) - (WINDOW_WIDTH // 2)
        location_y = (SCREEN.height // 2) - (WINDOW_HEIGHT // 2)
        self.set_location(location_x, location_y)

        self.background = pg.sprite.Sprite(background_img, x=0, y=0)
        
        self.car0 = CarObject(WINDOW_WIDTH/2, WINDOW_HEIGHT/2, car_img)
        self.line_detectors = LineDetectors(self.car0, True)
        self.distance_module = DistanceModule(self.car0, True)

        self.car1 = CarObject(WINDOW_WIDTH/2 + 33, WINDOW_HEIGHT/2, car_img)

        self.cars = [self.car0, self.car1]

    def on_draw(self):
        self.clear()
        self.background.draw()

        self.car0.draw()
        self.line_detectors.draw()
        self.distance_module.draw()
        self.car1.draw()

    def update(self, dt):
        self.car0.handleKeys()
        self.car0.update(dt)
        self.line_detectors.update(dt)
        self.distance_module.update(dt)

        print(self.distance_module.getData(self.cars))

        self.car1.update(dt)

if __name__ == "__main__":
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
