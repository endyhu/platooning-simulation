import ai
import simulation
import pyglet as pg

from pyglet.window import key

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

TITLE = "Platooning Simulator"

estimator = ai.Estimator()
estimator.load("_705.76_00057137_06_1452")

class Platoon:
    def __init__(self, distance):
        self.distance = distance
        self.first = None
        self.cars = []

    def addCar(self, car):
        if self.first == None:
            self.first = car
            car.max_velocity = 60.0
        else:
            car.optimal_distance = self.distance
    
        car.platoon = self
        self.cars.append(car)

if __name__ == "__main__":
    window = simulation.Window(WINDOW_WIDTH, WINDOW_HEIGHT, TITLE)
    window.push_handlers(simulation.KEYS)

    car0 = window.createCar(500, WINDOW_HEIGHT/2)
    car1 = window.createCar(500 - 33, WINDOW_HEIGHT/2)
    # car2 = window.createCar(500 - 66, WINDOW_HEIGHT/2)
    # car3 = window.createCar(500 - 99, WINDOW_HEIGHT/2)
    car4 = window.createCar(500 - 132, WINDOW_HEIGHT/2)
    car0.addEstimator(estimator)
    car1.addEstimator(estimator)
    # car2.addEstimator(estimator)
    # car3.addEstimator(estimator)
    car4.addEstimator(estimator)

    platoon0 = Platoon(11.0)
    platoon0.addCar(car0)
    platoon0.addCar(car1)
    # platoon0.addCar(car2)
    # platoon0.addCar(car3)
    platoon0.addCar(car4)

    pg.clock.schedule_interval(window.update, window.frame_rate)
    pg.app.run()
