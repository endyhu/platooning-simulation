import ai
import simulation
import pyglet as pg

from pyglet.window import key

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

TITLE = "Platooning Simulator"

estimator = ai.Estimator()
estimator.load("_705.76_00057137_06_1452")

if __name__ == "__main__":
    window = simulation.Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    window.push_handlers(simulation.KEYS)

    car0 = window.createCar(500, WINDOW_HEIGHT/2)
    car1 = window.createCar(500 - 33, WINDOW_HEIGHT/2)
    car0.addEstimator(estimator)
    car1.addEstimator(estimator)

    pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
