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
        self.pos_x = self.sprite.x
        self.pos_y = self.sprite.y

        self.acceleration = 0.0
        self.steering = 0
        self.velocity = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.max_acceleration = 20.0
        self.max_velocity = 40.0
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
                self.detector_sprites.append(pg.sprite.Sprite(sensor_img, 0, 0, batch=self.sprite_batch))

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
        acceleration = self.car.acceleration
        velocity = self.car.velocity
        line0, line1 = self.line_detectors.getData(True)

        return [acceleration, velocity, line0, line1]

    def reset(self):
        self.car.reset()

        return self.getState()

    def step(self, action):
        self.car.step(action)
        self.update(1/60)

        state = self.getState()
        reward = -0.1
        done = False

        if state[1] >= 30:
            reward = 1.0
        if state[2] or state[3]:
            reward = -1.0
        if state[2] and state[3]:
            reward = -1.0
            done = True

        return state, reward, done, None

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        print(self.step(1))

    def on_key_press(self, symbol, modifier):
        if symbol == key.R:
            self.reset()

    def on_draw(self):
        self.clear()
        self.background.draw()
        self.car.draw()
        self.line_detectors.draw()

    def update(self, dt):
        # self.car.handleKeys()
        self.car.update(dt)
        self.line_detectors.update(dt)

if __name__ == "__main__":
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    # pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
