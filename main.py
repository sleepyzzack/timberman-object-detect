import torch
from matplotlib import pyplot as pl
import os
import numpy as np
import cv2 as cv
import ipywidgets as widpy
import pyautogui as gui
import PIL.ImageGrab as imgrab
from tkinter import *
import keyboard as key
import time as t
import uuid as uid
import pymsgbox as msg

path = 'PROJECT1ml\yolov5-master'
img1 = 'PROJECT1ml\yolov5-master\data\images\zidane.jpg'
flag = True


model = torch.hub.load(path, 'custom', source='local', path='B:\Documentos\codigoXD\python\opencv\PROJECT1ml\yolov5-master\\runs\\train\exp8\weights\\best.pt' )

#esto sirve para resolver el error donde no se podia mostrar la imagen


result = model(img1)

pl.switch_backend('TkAgg')
pl.imshow(np.squeeze(result.render()))



#pl.show()
#pl.savefig("resultado.png")

def iniciar():
    while True:


        screen = gui.screenshot()




        screenArray = np.array(screen)

        y,x,c = screenArray.shape

        #print(f"y: {y}, x: {x}, c: {c}")
        #print(f"y: {y//2}, x: {x//2}, c: {c}")

        x1 = x//2
        y1 = y//2


        croppedRegion = screenArray[192:y1+350, 340:x1+500]


        coloresCorregidos = cv.cvtColor(croppedRegion, cv.COLOR_RGB2BGR)

        result2 = model(coloresCorregidos)




        t.sleep(0.5)

        print(result2)
        #cv.imshow('YOLO', np.squeeze(result2.render()))



        if key.is_pressed('¿'):
            break



def video():
    t.sleep(5)
    '''
    screen = gui.screenshot()

    screenArray = np.array(screen)

    y, x, c = screenArray.shape

    # print(f"y: {y}, x: {x}, c: {c}")
    # print(f"y: {y//2}, x: {x//2}, c: {c}")

    x1 = x // 2
    y1 = y // 2

    croppedRegion = screenArray[192:y1+550, 340:x1+500]
    coloresCorregidos = cv.cvtColor(croppedRegion, cv.COLOR_RGB2BGR)

    result2 = model(coloresCorregidos)

    cv.imshow('xd', np.squeeze(result2.render()))

    cv.waitKey(0)
    '''


    while True:

        # Specify resolution
        resolution = (1920, 1080)

        # Specify video codec
        codec = cv.VideoWriter_fourcc(*"XVID")

        # Specify name of Output file
        filename = "Recording.avi"

        # Specify frames rate. We can choose any
        # value and experiment with it in my case 60 fps works great
        fps = 60.0

        # Creating a VideoWriter object
        out = cv.VideoWriter(filename, codec, fps, resolution)

        # Create an Empty window
        cv.namedWindow("Live", cv.WINDOW_NORMAL)

        # Resize this window
        cv.resizeWindow("Live", 480, 270)




        # Take screenshot using PyAutoGUI 
        # there are many libraries that are useful but i gonna use this one xd
        img = gui.screenshot()

        # Convert the screenshot to a numpy array
        frame = np.array(img)

        # Convert it from BGR(Blue, Green, Red) to
        # RGB(Red, Green, Blue)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Write it to the output file
        #out.write(frame)
        result3 = model(frame)

        # Optional: Display the recording screen
        cv.imshow('Live', np.squeeze(result3.render()))


        # Stop recording when we press '.' or any key you wanna choose
        if cv.waitKey(1) == ord('.'):
            break

    # Release the Video writer
    out.release()

    # Destroy all windows
    cv.destroyAllWindows()

def recolectarImgs():




    IMAGES_PATH = os.path.join('datos', 'img')  # /data/images
    imgCantidad = int(msg.prompt(title="recoleccion de imagenes", text="escribe la cantidad de imagenes a recolectar"))
    t.sleep(3)

    for cantidad in range(imgCantidad):
        screen = gui.screenshot()

        screenArray = np.array(screen)

        y, x, c = screenArray.shape

        # print(f"y: {y}, x: {x}, c: {c}")
        # print(f"y: {y//2}, x: {x//2}, c: {c}")

        x1 = x // 2
        y1 = y // 2

        croppedRegion = screenArray[192:y1 + 350, 340:x1 + 500]


        imgname = os.path.join(IMAGES_PATH, str(uid.uuid1()) + '.jpg')
        coloresCorregidos = cv.cvtColor(croppedRegion, cv.COLOR_RGB2BGR)
        cv.imwrite(imgname, coloresCorregidos)

        result2 = model(coloresCorregidos)

    msg.alert(f"las {imgCantidad} imagenes han sido tomadas")

    menu()


def dosProbeMod():

    t.sleep(4)

    screen = gui.screenshot()

    screenArray = np.array(screen)

    y, x, c = screenArray.shape

    # print(f"y: {y}, x: {x}, c: {c}")
    # print(f"y: {y//2}, x: {x//2}, c: {c}")

    x1 = x // 2
    y1 = y // 2

    croppedRegion = screenArray[192:y1 + 350, 340:x1 + 500]

    coloresCorregidos = cv.cvtColor(croppedRegion, cv.COLOR_RGB2BGR)

    result2 = model(coloresCorregidos)

    cv.imshow('YOLOProbeMode', np.squeeze(result2.render()))
    cv.waitKey(0)
    menu()

def opciones(opcion):


    if opcion == "1":

        msg.alert(title="xd", text="en preceso de construccion xd")

    elif opcion == "2":

        recolectarImgs()

    elif opcion == "3":

        iniciar()

    elif opcion == "1 probmode":

        msg.alert(title="xd", text="en preceso de construccion xd")

    elif opcion == "2 probmode":

        dosProbeMod()

    elif opcion == "vid":
        video()


    elif opcion == None:

        exit()

    else:

        menu()




        
    return opcion



def menu():
    opcion = msg.prompt(title="escribe el numero de una opcion",
                        text="que quieres hacer?, 1- entrenar, 2- recolectar imagenes, 3- ejecutar bot")

    if opcion != "":

        opciones(opcion)
    elif opcion == None:
        exit()


    else:
        msg.alert(text="escribe una opcion")
        menu()



try:

    t.sleep(1)

    menu()

except KeyboardInterrupt:

    exit()








