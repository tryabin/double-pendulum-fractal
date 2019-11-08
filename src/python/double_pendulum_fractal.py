from math import *
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import os
import time
from win32api import GetSystemMetrics
from utils import read_file

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule



class DoublePendulumFractalApp(tk.Tk):

    # The range of angles.
    angle1Min = 1/2*pi
    angle1Max = 3/2*pi
    angle2Min = 0
    angle2Max = 2*pi

    # Number of pixels for the X and Y axes.
    numberOfAnglesToTestX = int(500 / 2**0)
    numberOfAnglesToTestY = int(1000 / 2**0)

    # Simulation parameters.
    timestep = .001
    maxTimeToSeeIfPendulumFlips = 2**5
    zoomFactor = 2
    numberOfAnglesToTestPerKernelCallRatio = 1

    # Pendulum model parameters.
    gravity = 9.81
    point1Mass = 1
    point2Mass = 1
    pendulum1Length = 1
    pendulum2Length = 1

    # Initialize the kernel.
    useDoublePrecision = False  # Enabling double precision could slow performance by 20 times or more.
    includeDir = os.getcwd() + '/include'
    kernelFile = 'src/cuda/double_pendulum_fractal.cu' if not useDoublePrecision else 'src/cuda/double_pendulum_fractal_double_precision.cu' 
    doublePendulumFractalFunction = SourceModule(read_file(kernelFile), include_dirs=[includeDir]).get_function('compute_double_pendulum_fractal_image')

    def __init__(self):
        tk.Tk.__init__(self)

        # Set the window size and position.
        self.geometry('%dx%d+%d+%d' % (self.numberOfAnglesToTestX,
                                       self.numberOfAnglesToTestY,
                                       GetSystemMetrics(0)/2 - self.numberOfAnglesToTestX/2,
                                       GetSystemMetrics(1)/2 - self.numberOfAnglesToTestY/2))

        self.canvas = tk.Canvas(self, width=self.numberOfAnglesToTestX, height=self.numberOfAnglesToTestY)
        self.canvas.pack(side='top', fill='both', expand=True)
        self.canvas.bind('z', self.zoom_in)
        self.canvas.bind('x', self.zoom_out)
        self.canvas.focus_set()
        
        self.draw_fractal()


    def zoom_in(self, event):
        center1 = (event.x / self.numberOfAnglesToTestX) * (self.angle1Max - self.angle1Min) + self.angle1Min
        newWidth = (self.angle1Max - self.angle1Min) / self.zoomFactor
        self.angle1Min = center1 -  newWidth/2
        self.angle1Max = center1 + newWidth/2

        center2 = (1 - event.y / self.numberOfAnglesToTestY) * (self.angle2Max - self.angle2Min) + self.angle2Min
        newHeight = (self.angle2Max - self.angle2Min) / self.zoomFactor
        self.angle2Min = center2 - newHeight / 2
        self.angle2Max = center2 + newHeight / 2

        print('angle1Min = ' + str(self.angle1Min))
        print('angle1Max = ' + str(self.angle1Max))
        print('angle2Min = ' + str(self.angle2Min))
        print('angle2Max = ' + str(self.angle2Max))

        self.draw_fractal()
        

    def zoom_out(self, event):
        center1 = (event.x / self.numberOfAnglesToTestX) * (self.angle1Max - self.angle1Min) + self.angle1Min
        newWidth = (self.angle1Max - self.angle1Min) * self.zoomFactor
        self.angle1Min = center1 -  newWidth/2
        self.angle1Max = center1 + newWidth/2

        center2 = (1 - event.y / self.numberOfAnglesToTestY) * (self.angle2Max - self.angle2Min) + self.angle2Min
        newHeight = (self.angle2Max - self.angle2Min) * self.zoomFactor
        self.angle2Min = center2 - newHeight / 2
        self.angle2Max = center2 + newHeight / 2

        self.draw_fractal()


    def draw_fractal(self):

        print('Drawing fractal')

        # Compute the double pendulum fractal image.
        start = time.time()

        colors = np.zeros((3, self.numberOfAnglesToTestY, self.numberOfAnglesToTestX), np.dtype(np.uint8))
        for i in range(self.numberOfAnglesToTestPerKernelCallRatio):
            for j in range(self.numberOfAnglesToTestPerKernelCallRatio):

                print('Running kernel...')

                kernelStart = time.time()
                curColors = np.zeros_like(colors)
                
                if not self.useDoublePrecision:
                    self.doublePendulumFractalFunction(np.float32(self.point1Mass), np.float32(self.point2Mass),
                                                       np.float32(self.pendulum1Length), np.float32(self.pendulum2Length),
                                                       np.float32(self.gravity),
                                                       np.float32(self.angle1Min), np.float32(self.angle1Max),
                                                       np.float32(self.angle2Min), np.float32(self.angle2Max),
                                                       np.int32(self.numberOfAnglesToTestPerKernelCallRatio),
                                                       np.int32(i), np.int32(j),
                                                       np.int32(self.numberOfAnglesToTestX), np.int32(self.numberOfAnglesToTestY),
                                                       np.float32(self.timestep),
                                                       np.float32(self.maxTimeToSeeIfPendulumFlips),
                                                       drv.Out(curColors),
                                                       # colorsGpu,
                                                       # block=(4, 4, 1), grid=(4, 4))
                                                       # block=(8, 8, 1), grid=(8, 8))
                                                       block=(16, 16, 1), grid=(16, 16))
                                                # block=(32, 32, 1), grid=(32, 32))
                                                # block=(2, 2, 1), grid=(1, 1))
                
                else:
                    self.doublePendulumFractalFunction(np.float64(self.point1Mass), np.float64(self.point2Mass),
                                                       np.float64(self.pendulum1Length), np.float64(self.pendulum2Length),
                                                       np.float64(self.gravity),
                                                       np.float64(self.angle1Min), np.float64(self.angle1Max),
                                                       np.float64(self.angle2Min), np.float64(self.angle2Max),
                                                       np.int32(self.numberOfAnglesToTestPerKernelCallRatio),
                                                       np.int32(i), np.int32(j),
                                                       np.int32(self.numberOfAnglesToTestX), np.int32(self.numberOfAnglesToTestY),
                                                       np.float64(self.timestep),
                                                       np.float64(self.maxTimeToSeeIfPendulumFlips),
                                                       drv.Out(curColors),
                                                       # colorsGpu,
                                                       # block=(4, 4, 1), grid=(4, 4))
                                                       # block=(8, 8, 1), grid=(8, 8))
                                                       block=(16, 16, 1), grid=(16, 16))
                                                # block=(32, 32, 1), grid=(32, 32))
                                                # block=(2, 2, 1), grid=(1, 1))


                colors = np.add(colors, curColors)
                
                print('Completed ' + str(int(i*self.numberOfAnglesToTestPerKernelCallRatio + j + 1)) + ' out of ' +
                      str(int(pow(self.numberOfAnglesToTestPerKernelCallRatio, 2))) + ' kernels in ' + str(time.time() - kernelStart) + ' seconds')



        print('total time to run kernel = ' + str(time.time() - start))

        # Display the image.
        redArray = Image.fromarray(colors[0])
        greenArray = Image.fromarray(colors[1])
        blueArray = Image.fromarray(colors[2])

        pilImageRGB = Image.merge('RGB', (redArray, greenArray, blueArray))
        img = ImageTk.PhotoImage(pilImageRGB)
        self.canvas.create_image((self.numberOfAnglesToTestX / 2, self.numberOfAnglesToTestY / 2), image=img, state="normal")

        # Save the image so it isn't garbage collected.
        self.image = img

        # Save the image to a file.
        filename = "double pendulum fractal.png"
        pilImageRGB.save(filename)



if __name__ == "__main__":
    app = DoublePendulumFractalApp()
    app.mainloop()

