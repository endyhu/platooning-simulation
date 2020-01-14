import ai
import time
import json
import requests
import numpy as np

url = "http://192.168.43.92:3000/api/cars/"

def getData(id=""):
    global url
    request_data = requests.get(url+str(id), verify=False)
    return request_data.json()

def updateData(id, leftspeed, rightspeed, leftlinesensor, rightlinesensor, ultrasonicsensor):
    global url
    payload = {"id": id, "leftspeed": leftspeed, "rightspeed": rightspeed, "leftlinesensor": leftlinesensor,
               "rightlinesensor": rightlinesensor, "ultrasonicsensor": ultrasonicsensor}
    headers = {'content-type': 'application/json'}
    r = requests.put(url, data=json.dumps(payload),
                     headers=headers, verify=False)
    return r.status_code

def main():
    estimator = ai.Estimator()
    estimator.load("_705.76_00057137_06_1452")

    acceleration = 0
    max_acceleration = 10.0
    velocity = 0
    max_velocity = 40.0

    while True:
        output_data = [i for i in getData(1).values()]
        output_data.pop(0)

        output_data[0] = acceleration
        output_data[1] = velocity

        prediction = estimator.predict(output_data)
        action = np.argmax(prediction)

        ratio_l = 1
        ratio_r = 1
        acceleration = -max_acceleration
        if action == 1:
            acceleration = max_acceleration
        elif action == 4:
            # steering = -1
            ratio_l = 0.0
            ratio_r = 1.0
            acceleration = max_acceleration * ratio_l - max_acceleration * ratio_r
        elif action == 3:
            acceleration = -max_acceleration
        elif action == 2:
            # steering = 1
            ratio_l = 1.0
            ratio_r = 0.0
            acceleration = max_acceleration * ratio_l - max_acceleration * ratio_r
        elif action == 6:
            # steering = -1
            ratio_l = 0.25
            ratio_r = 1.0
            acceleration = max_acceleration * ratio_l - max_acceleration * ratio_r
        elif action == 5:
            # steering = 1
            ratio_l = 1.0
            ratio_r = 0.25
            acceleration = max_acceleration * ratio_l - max_acceleration * ratio_r
        
        velocity = velocity + acceleration
        if velocity > max_velocity:
            velocity = max_velocity
        elif velocity < 0:
            velocity = 0

        updateData(1, velocity*ratio_l, velocity*ratio_r, output_data[2], output_data[3], output_data[4])

        print(f"Action: {action} Acceleration: {acceleration} Velocity: {velocity}")

        time.sleep(0.1)

main()
