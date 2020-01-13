import pygame
import requests
import json

url = "http://192.168.43.92:3000/api/cars/"
pygame.init()

win = pygame.display.set_mode((500,500))
pygame.display.set_caption("First Game")


run = True

def getData(id=""):
    global url
    request_data = requests.get(url+str(id), verify=False)
    return request_data.json()

def updateData(id,leftspeed,rightspeed,leftlinesensor,rightlinesensor,ultrasonicsensor):
    global url
    payload = {"id":id,"leftspeed":leftspeed,"rightspeed":rightspeed,"leftlinesensor":leftlinesensor,"rightlinesensor":rightlinesensor,"ultrasonicsensor":ultrasonicsensor}
    headers = {'content-type': 'application/json'}
    r = requests.put(url, data=json.dumps(payload), headers=headers, verify=False)
    return r.status_code

while run:
    pygame.time.delay(100)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_LEFT]:
        print("left")
        updateData(1,50,0,False,False,0)

    if keys[pygame.K_RIGHT]:
        print("right")
        
        updateData(1,0,50,False,False,0)

    if keys[pygame.K_UP]:
        print("up")
        updateData(1,100,100,False,False,0)

    if keys[pygame.K_DOWN]:
        print("down")
        updateData(1,-100,-100,False,False,0)

    if keys[pygame.K_p]:
        print("stand still")
        updateData(1,0,0,False,False,0)
    
    win.fill((0,0,0))  # Fills the screen with black
    
    pygame.display.update() 
    
pygame.quit()