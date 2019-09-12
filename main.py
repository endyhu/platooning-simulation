import numpy as np
import pyglet as pg

from pyglet.window import key

PLATFORM = pg.window.get_platform()
DISPLAY = PLATFORM.get_default_display()
SCREEN = DISPLAY.get_default_screen()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

TITLE = "Platooning Simulator"

background_img = pg.image.load("./assets/grass.png")
road_top_img = pg.image.load("./assets/road_top.png")
road_mid_img = pg.image.load("./assets/road_mid.png")
road_bot_img = pg.image.load("./assets/road_bot.png")
car_img = pg.image.load("./assets/car.png")

def centerImage(image):
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2

def centerVehicleImage(image):
    # image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2

centerVehicleImage(car_img)

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

class RoadObject(Object):
    def __init__(self, lanes=3):
        self.lanes = lanes
        self.lane_width = road_mid_img.height

        self.sprite_batch = pg.graphics.Batch()
        self.road_sprites = []

        for i in range(8):
            pos_x = i * 120
            pos_y = (WINDOW_HEIGHT / 2) - (self.lane_width / 2) + (self.lane_width * (self.lanes - 1) / 2)
            for j in range(self.lanes):
                road_img = road_mid_img
                
                if j != 0:
                    pos_y = pos_y - self.lane_width
                if j == 0:
                    road_img = road_top_img
                elif j == self.lanes - 1:
                    road_img = road_bot_img
                    
                self.road_sprites.append(pg.sprite.Sprite(road_img, pos_x, pos_y, batch=self.sprite_batch))

    def draw(self):
        self.sprite_batch.draw()

    def update(self, velocity, dt):
        for i, sprite in enumerate(self.road_sprites):
            sprite.x = sprite.x - velocity * dt
            
            if sprite.x < -120:
                sprite.x = sprite.x + (120 * 8)
            elif sprite.x > (120 * 7):
                sprite.x = sprite.x - (120 * 8)

class CarObject(Object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = self.sprite.width

        self.steering = 0
        self.tyre_angle = 0.0
        self.acceleration = 0.0
        self.velocity = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.nrm_deceleration = 25.0
        self.brk_deceleration = 100.0

        self.max_tyre_angle = 25.0
        self.max_acceleration = 59.1
        self.max_velocity = 277.8
        self.max_velocity_r = -138.9

    def handleKeys(self):
        if keys[key.W]:
            self.acceleration = self.max_acceleration
            if self.velocity < 0.0:
                self.acceleration = self.brk_deceleration
        if keys[key.A]:
            self.steering = -1
        if keys[key.S]:
            self.acceleration = -self.max_acceleration * 0.8
            if self.velocity > 0.0:
                self.acceleration = -self.brk_deceleration
        if keys[key.D]:
            self.steering = 1
        if keys[key.SPACE]:
            self.acceleration = -np.sign(self.velocity) * self.brk_deceleration

        if keys[key.W] and keys[key.S]:
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
                self.tyre_angle = self.tyre_angle - (np.sign(self.tyre_angle) * self.max_tyre_angle * 3) * dt
        else:
            self.tyre_angle = self.tyre_angle + (self.steering * self.max_tyre_angle * 3) * dt

        if self.tyre_angle != 0.0:
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

        self.velocity_x = self.velocity * np.cos(np.deg2rad(self.sprite.rotation))
        self.velocity_y = self.velocity * -np.sin(np.deg2rad(self.sprite.rotation))

        # self.sprite.x = self.sprite.x + self.velocity_x * dt
        self.sprite.y = self.sprite.y + self.velocity_y * dt

class Window(pg.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_rate = 1/60.0

        location_x = (SCREEN.width // 2) - (WINDOW_WIDTH // 2)
        location_y = (SCREEN.height // 2) - (WINDOW_HEIGHT // 2)
        self.set_location(location_x, location_y)

        self.background = pg.sprite.Sprite(background_img, x=0, y=0)
        
        self.road = RoadObject(5)
        self.car = CarObject(WINDOW_WIDTH/2, WINDOW_HEIGHT/2, car_img)

    def on_key_release(self, symbol, modifiers):
        self.car.on_key_release(symbol, modifiers)

    def on_draw(self):
        self.clear()
        self.background.draw()
        self.road.draw()
        self.car.draw()

    def update(self, dt):
        self.road.update(self.car.velocity_x, dt)
        self.car.handleKeys()
        self.car.update(dt)

if __name__ == "__main__":
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
