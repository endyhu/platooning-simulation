import ai
import time
import json
import requests

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

    while True:
        output_data = [i for i in getData(1).values()]
        output_data.pop(0)

        prediction = estimator.predict(output_data)
        print(prediction)

        time.sleep(1)

main()
