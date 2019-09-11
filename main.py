import pyglet
import numpy as np

from pyglet.window import key

PLATFORM = pyglet.window.get_platform()
DISPLAY = PLATFORM.get_default_display()
SCREEN = DISPLAY.get_default_screen()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

TITLE = "Platooning Simulator"

background_img = pyglet.image.load("./assets/grass.png")
car_img = pyglet.image.load("./assets/car.png")

def centerImage(image):
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2

def centerVehicleImage(image):
    image.anchor_x = image.width // 5
    image.anchor_y = image.height // 2

centerVehicleImage(car_img)

class Object:
    def __init__(self, pos_x, pos_y, image=None):
        self.pos_x = pos_x
        self.pos_y = pos_y

        if image is not None:
            self.sprite = pyglet.sprite.Sprite(image, pos_x, pos_y)

    def draw(self):
        self.sprite.draw()

    def update(self, dt):
        pass

class CarObject(Object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = self.sprite.width

        self.steering = 0
        self.tyre_angle = 0.0
        self.acceleration = 0.0
        self.velocity = 0.0

        self.nrm_deceleration = 25.0
        self.brk_deceleration = 100.0

        self.max_tyre_angle = 30.0
        self.max_acceleration = 59.1
        self.max_velocity = 277.8
        self.max_velocity_r = -166.7

    def handleKeys(self):
        if keys[key.W]:
            self.acceleration = self.max_acceleration
        if keys[key.A]:
            self.steering = -10
        if keys[key.S]:
            self.acceleration = -self.max_acceleration * 0.8
        if keys[key.D]:
            self.steering = 10
        if keys[key.SPACE]:
            self.acceleration = -np.sign(self.velocity) * self.brk_deceleration

        if keys[key.A] and keys[key.D]:
            self.steering = 0

    def on_key_release(self, symbol, modifiers):
        if symbol == key.W:
            self.acceleration = 0
        if symbol == key.A:
            self.steering = 0
        if symbol == key.S:
            self.acceleration = 0
        if symbol == key.D:
            self.steering = 0
        if symbol == key.SPACE:
            self.acceleration = 0

    def update(self, dt):
        if abs(self.tyre_angle) > self.max_tyre_angle:
            self.tyre_angle = np.sign(self.tyre_angle) * self.max_tyre_angle
        elif self.steering == 0:
            if abs(self.tyre_angle) < 5.0:
                self.tyre_angle = 0.0
            else:
                self.tyre_angle = self.tyre_angle - (np.sign(self.tyre_angle) * 10 * self.max_tyre_angle) * dt
        else:
            self.tyre_angle = self.tyre_angle + (self.steering * self.max_tyre_angle) * dt

        if self.tyre_angle:
            turning_radius = self.length / np.tan(np.deg2rad(self.tyre_angle))
            angular_velocity = self.velocity / turning_radius
        else:
            angular_velocity = 0.0

        self.sprite.rotation = self.sprite.rotation + np.rad2deg(angular_velocity) * dt

        if self.velocity > self.max_velocity:
            self.velocity = self.max_velocity
        elif self.velocity < self.max_velocity_r:
            self.velocity = self.max_velocity_r
        elif self.acceleration == 0.0:
            if abs(self.velocity) < 5.0:
                self.velocity = 0.0
            else:
                self.velocity = self.velocity - (np.sign(self.velocity) * self.nrm_deceleration) * dt
        else:
            self.velocity = self.velocity + self.acceleration * dt

        velocity_x = np.cos(np.deg2rad(self.sprite.rotation))
        velocity_y = -np.sin(np.deg2rad(self.sprite.rotation))

        self.sprite.x = self.sprite.x + self.velocity * velocity_x * dt
        self.sprite.y = self.sprite.y + self.velocity * velocity_y * dt

class Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_rate = 1/60.0

        location_x = (SCREEN.width // 2) - (WINDOW_WIDTH // 2)
        location_y = (SCREEN.height // 2) - (WINDOW_HEIGHT // 2)
        self.set_location(location_x, location_y)

        self.background = pyglet.sprite.Sprite(background_img, x=0, y=0)

        self.car = CarObject(300, 300, car_img)

    # def on_key_press(self, symbol, modifiers):
        # image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data().data
        # image_data = np.frombuffer(image_data, dtype=np.uint8).reshape(WINDOW_HEIGHT, WINDOW_WIDTH, 4)
        # image_data = image_data[:, :, :-1]

    def on_key_release(self, symbol, modifiers):
        self.car.on_key_release(symbol, modifiers)

    def on_draw(self):
        self.clear()
        self.background.draw()
        self.car.draw()

    def update(self, dt):
        self.car.handleKeys()
        self.car.update(dt)

if __name__ == "__main__":
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    pyglet.clock.schedule_interval(window.update, window.frame_rate)
    pyglet.app.run()
