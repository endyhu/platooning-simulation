import simulation
import pyglet as pg

from pyglet.window import key

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

TITLE = "Platooning Simulator"

if __name__ == "__main__":
    window = simulation.Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    window.push_handlers(simulation.KEYS)

    platoon0 = window.createPlatoon(11.0)

    pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
