from math import *
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import os
import time

from double_pendulum import get_point_position
from numerical_routines import compute_double_pendulum_step_rk4
from utils import read_file

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule




def main():
    angle1Min = -pi
    angle1Max = pi
    angle2Min = -pi
    angle2Max = pi
    numberOfAnglesToTest = 1000
    timestep = .01
    maxTimeToSeeIfPendulumFlips = 20

    gravity = 9.81
    point1Mass = 10
    point2Mass = 10
    pendulum1Length = 1
    pendulum2Length = 1

    colors = np.zeros((3, numberOfAnglesToTest, numberOfAnglesToTest), np.dtype(np.uint8))


    # Initialize the kernel.
    includeDir = os.getcwd() + '/include'
    doublePendulumFunction = SourceModule(read_file('src/cuda/double_pendulum_fractal.cu'), include_dirs=[includeDir]).get_function('compute_double_pendulum_fractal_image')

    start = time.time()
    # Compute the double pendulum fractal image.
    doublePendulumFunction(np.float32(point1Mass), np.float32(point2Mass),
                           np.float32(pendulum1Length), np.float32(pendulum2Length),
                           np.float32(gravity),
                           np.float32(angle1Min), np.float32(angle1Max),
                           np.float32(angle2Min), np.float32(angle2Max),
                           np.int32(numberOfAnglesToTest),
                           np.float32(timestep),
                           np.float32(maxTimeToSeeIfPendulumFlips),
                           drv.Out(colors),
                           block=(16, 16, 1), grid=(16, 16))
                           # block=(32, 32, 1), grid=(32, 32))

    print('kernel time = ' + str(time.time() - start))


    '''
    print('\n\n\n')

    for i in range(numberOfAnglesToTest):
        for j in range(numberOfAnglesToTest):
                
            # print('Computing (' + str(i) + ', ' + str(j) + ')')

            curAngle1 = angle1Min + i*(angle1Max - angle1Min)/numberOfAnglesToTest
            curAngle2 = angle2Min + j*(angle2Max - angle2Min)/numberOfAnglesToTest

            point1AngularVelocity = 0
            point2AngularVelocity = 0
            
            curTime = 0
            while curTime < maxTimeToSeeIfPendulumFlips:

                point1OriginalPosition = get_point_position([0,0], curAngle1, pendulum1Length)

                if i == 4 and j == 4:
                    print('point1OriginalPosition X = ' + str(point1OriginalPosition[0]))
                    print('point1OriginalPosition Y = ' + str(point1OriginalPosition[1]))

                point1AngularVelocity, point2AngularVelocity, curAngle1, curAngle2 = compute_double_pendulum_step_rk4(point1Mass, point2Mass,
                                                                                                                      gravity,
                                                                                                                      pendulum1Length, pendulum2Length,
                                                                                                                      point1AngularVelocity, point2AngularVelocity,
                                                                                                                      curAngle1, curAngle2,
                                                                                                                      timestep)

                point1CurrentPosition = get_point_position([0,0], curAngle1, pendulum1Length)
                
                # Check to see if the first mass flipped.
                curTime += timestep
                if point1CurrentPosition[0]*point1OriginalPosition[0] < 0 < point1CurrentPosition[1]:
                    break




            # Compute the color for the pixel for the current initial position.
            milliseconds = curTime*1000
            shift = 1.1
            r = 1.0
            g = 4.0
            b = 7.2
            colors[0][i][j] = abs(sin(1.0/255 * pi * milliseconds * r * shift)) * 255
            colors[1][i][j] = abs(sin(1.0/255 * pi * milliseconds * g * shift)) * 255
            colors[2][i][j] = abs(sin(1.0/255 * pi * milliseconds * b * shift)) * 255

    '''
    
    # Display the image.
    redArray = Image.fromarray(colors[0])
    greenArray = Image.fromarray(colors[1])
    blueArray = Image.fromarray(colors[2])

    root = tk.Tk()
    pilImageRGB = Image.merge('RGB', (redArray, greenArray, blueArray))
    img = ImageTk.PhotoImage(pilImageRGB)
    canvas = tk.Canvas(root, width=numberOfAnglesToTest, height=numberOfAnglesToTest)
    canvas.pack(side='top', fill='both', expand=True)
    canvas.create_image((numberOfAnglesToTest / 2, numberOfAnglesToTest / 2), image=img, state="normal")

    tk.mainloop()





if __name__ == "__main__":
    main()

