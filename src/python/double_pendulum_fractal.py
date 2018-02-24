from math import *
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import os
import time
from utils import read_file

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule




class DoublePendulumFractalApp(tk.Tk):

    angle1Min = -pi
    angle1Max = pi
    angle2Min = -pi
    angle2Max = pi
    numberOfAnglesToTestPerKernelCallRatio = 4
    numberOfAnglesToTestX = 1000
    numberOfAnglesToTestY = 1000
    timestep = .001
    maxTimeToSeeIfPendulumFlips = 8
    zoomFactor = 2

    gravity = 9.81
    point1Mass = 10
    point2Mass = 10
    pendulum1Length = 1
    pendulum2Length = 1

    colors = np.zeros((3, numberOfAnglesToTestY, numberOfAnglesToTestX), np.dtype(np.uint8))

    # Initialize the kernel.
    includeDir = os.getcwd() + '/include'
    doublePendulumFunction = SourceModule(read_file('src/cuda/double_pendulum_fractal.cu'), include_dirs=[includeDir]).get_function('compute_double_pendulum_fractal_image')

    def __init__(self):
        tk.Tk.__init__(self)

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

        print('self.angle1Min = ' + str(self.angle1Min))
        print('self.angle1Max = ' + str(self.angle1Max))
        print('self.angle2Min = ' + str(self.angle2Min))
        print('self.angle2Max = ' + str(self.angle2Max))

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
        colorsGpu = drv.mem_alloc(self.colors.nbytes)

        start = time.time()

        for i in range(self.numberOfAnglesToTestPerKernelCallRatio):
            for j in range(self.numberOfAnglesToTestPerKernelCallRatio):

                self.doublePendulumFunction(np.float32(self.point1Mass), np.float32(self.point2Mass),
                                            np.float32(self.pendulum1Length), np.float32(self.pendulum2Length),
                                            np.float32(self.gravity),
                                            np.float32(self.angle1Min), np.float32(self.angle1Max),
                                            np.float32(self.angle2Min), np.float32(self.angle2Max),
                                            np.int32(self.numberOfAnglesToTestPerKernelCallRatio),
                                            np.int32(i), np.int32(j),
                                            np.int32(self.numberOfAnglesToTestX), np.int32(self.numberOfAnglesToTestY),
                                            np.float32(self.timestep),
                                            np.float32(self.maxTimeToSeeIfPendulumFlips),
                                            colorsGpu,
                                            # block=(4, 4, 1), grid=(4, 4))
                                            # block=(8, 8, 1), grid=(8, 8))
                                            block=(16, 16, 1), grid=(16, 16))
                                            # block=(32, 32, 1), grid=(32, 32))


                drv.Context.synchronize()
                print('Completed ' + str(int(i*self.numberOfAnglesToTestPerKernelCallRatio + j + 1)) + ' out of ' +  str(int(pow(self.numberOfAnglesToTestPerKernelCallRatio, 2))) + ' kernels')


        print('total kernel time = ' + str(time.time() - start))

        start = time.time()
        drv.memcpy_dtoh(self.colors, colorsGpu)

        print('copy time = ' + str(time.time() - start))

        # Display the image.
        redArray = Image.fromarray(self.colors[0])
        greenArray = Image.fromarray(self.colors[1])
        blueArray = Image.fromarray(self.colors[2])

        pilImageRGB = Image.merge('RGB', (redArray, greenArray, blueArray))
        img = ImageTk.PhotoImage(pilImageRGB)

        self.canvas.create_image((self.numberOfAnglesToTestX / 2, self.numberOfAnglesToTestY / 2), image=img, state="normal")

        # Save the image so it isn't garbage collected.
        self.image = img



if __name__ == "__main__":
    app = DoublePendulumFractalApp()
    app.mainloop()

