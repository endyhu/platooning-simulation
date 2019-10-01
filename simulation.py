import cv2
import numpy as np
import pyglet as pg

from pyglet.window import key

DISPLAY = pg.canvas.get_display()
SCREEN = DISPLAY.get_default_screen()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

TITLE = "Platooning Simulator"

background_img = pg.image.load("./assets/grass.png")
car_img = pg.image.load("./assets/debug_car.png")
sensor_img = pg.image.load("./assets/sensor.png")

background_data = cv2.imread("./assets/grass.png")

def centerImage(image):
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2

def centerCarImage(image):
    image.anchor_x = image.width // 3
    image.anchor_y = image.height // 2

centerCarImage(car_img)

class Object:
    def __init__(self, pos_x, pos_y, image=None):
        self.pos_x = pos_x
        self.pos_y = pos_y

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
            self.velocity = 0.0
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

class LineDetectorLeft(Object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, dt, car):
        self.sprite.x = car.x + (car.width / 3 * 2) * np.cos(np.deg2rad(car.rotation)) - (car.height / 2) * -np.sin(np.deg2rad(car.rotation))
        self.sprite.y = car.y + (car.width / 3 * 2) * -np.sin(np.deg2rad(car.rotation)) + (car.height / 2) * np.cos(np.deg2rad(car.rotation))

class LineDetectorRight(Object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, dt, car):
        self.sprite.x = car.x + (car.width / 3 * 2) * np.cos(np.deg2rad(car.rotation)) + (car.height / 2) * -np.sin(np.deg2rad(car.rotation))
        self.sprite.y = car.y + (car.width / 3 * 2) * -np.sin(np.deg2rad(car.rotation)) - (car.height / 2) * np.cos(np.deg2rad(car.rotation))

class Window(pg.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_rate = 1/60.0

        location_x = (SCREEN.width // 2) - (WINDOW_WIDTH // 2)
        location_y = (SCREEN.height // 2) - (WINDOW_HEIGHT // 2)
        self.set_location(location_x, location_y)

        self.background = pg.sprite.Sprite(background_img, x=0, y=0)
        
        self.car = CarObject(WINDOW_WIDTH/2, WINDOW_HEIGHT/2, car_img)
        self.line_detector_left = LineDetectorLeft(0, 0, sensor_img)
        self.line_detector_right = LineDetectorRight(0, 0, sensor_img)

    def on_draw(self):
        self.clear()
        self.background.draw()
        self.car.draw()
        self.line_detector_left.draw()
        self.line_detector_right.draw()

    def update(self, dt):
        self.car.handleKeys()
        self.car.update(dt)
        self.line_detector_left.update(dt, self.car.sprite)
        self.line_detector_right.update(dt, self.car.sprite)

if __name__ == "__main__":
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
