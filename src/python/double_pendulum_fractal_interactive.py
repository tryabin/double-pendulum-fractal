import logging
import os
import sys
import time
import tkinter as tk
from math import *

import numpy as np
from PIL import ImageTk
from win32api import GetSystemMetrics

from double_pendulum_kernel_methods import DoublePendulumCudaSimulator
from utils import save_image_to_file

logger = logging.getLogger('root')

class DoublePendulumFractalApp(tk.Tk):
    # Control parameters.
    zoomFactor = 2
    maxTimeToSeeIfPendulumFlipsChangeFactor = 2
    timeStepFactor = 2
    maxTimeToSeeIfPendulumFlipsSeconds = 2**5

    def __init__(self):
        tk.Tk.__init__(self)

        # Initialize the logger.
        self.directoryToSaveData = './interactive'
        if not os.path.exists(self.directoryToSaveData):
            os.mkdir(self.directoryToSaveData)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(self.directoryToSaveData + '/log.log', mode='w'))

        # Initialize the simulator.
        deviceNumberToUse = 0 # Determines which GPU is used to run the simulation.
        useDoublePrecision = False
        self.simulator = DoublePendulumCudaSimulator(deviceNumberToUse, self.directoryToSaveData, useDoublePrecision)

        # The range of pendulum angles.
        self.simulator.set_angle1_min(-3/2*pi)
        self.simulator.set_angle1_max(-1/2*pi)
        self.simulator.set_angle2_min(0*pi)
        self.simulator.set_angle2_max(2*pi)
        # self.simulator.set_angle1_min(-3.396454357612266)
        # self.simulator.set_angle1_max(-3.371910665006095)
        # self.simulator.set_angle2_min(1.901448953585222)
        # self.simulator.set_angle2_max(1.925992646191392)

        # The width of the image in pixels.
        self.simulator.set_image_width_pixels(500)

        # The amount of super-sampling anti-aliasing to apply to the image. Can be fractional.
        # 1 means no anti-aliasing.
        # 2 means four total samples are used per pixel.
        # 3 means nine total samples are used per pixel, etc.
        self.simulator.set_anti_aliasing_amount(2)

        # Simulation parameters.
        self.simulator.set_time_step(.01/2**2)
        self.simulator.set_gravity(1)
        self.simulator.set_point1_mass(1)
        self.simulator.set_point2_mass(1)
        self.simulator.set_pendulum1_length(1)
        self.simulator.set_pendulum2_length(1)

        # Initialize the data containing the pendulum states and the number of time steps before the pendulums flip.
        self.initialize_data()

        # Set the window size and position.
        self.geometry('%dx%d+%d+%d' % (self.simulator.imageResolutionPixelsWidth,
                                       self.simulator.imageResolutionPixelsHeight,
                                       GetSystemMetrics(0)/2 - self.simulator.imageResolutionPixelsWidth/2,
                                       GetSystemMetrics(1)/2 - self.simulator.imageResolutionPixelsHeight/2))

        self.canvas = tk.Canvas(self, width=self.simulator.imageResolutionPixelsWidth, height=self.simulator.imageResolutionPixelsHeight)
        self.canvas.pack(side='top', fill='both', expand=True)
        self.canvas.bind('z', self.zoom_in)
        self.canvas.bind('x', self.zoom_out)
        self.canvas.bind('a', self.increase_time_to_wait_for_flip)
        self.canvas.bind('s', self.decrease_time_to_wait_for_flip)
        self.canvas.bind('q', self.decrease_time_step)
        self.canvas.bind('w', self.increase_time_step)
        self.canvas.focus_set()
        
        self.draw_fractal(True)


    def zoom_in(self, event):
        center1 = (event.x / self.simulator.imageResolutionPixelsWidth) * (self.simulator.angle1Max - self.simulator.angle1Min) + self.simulator.angle1Min
        newWidth = (self.simulator.angle1Max - self.simulator.angle1Min) / self.zoomFactor
        self.simulator.set_angle1_min(center1 -  newWidth/2)
        self.simulator.set_angle1_max(center1 + newWidth/2)

        center2 = (1 - event.y / self.simulator.imageResolutionPixelsHeight) * (self.simulator.angle2Max - self.simulator.angle2Min) + self.simulator.angle2Min
        newHeight = (self.simulator.angle2Max - self.simulator.angle2Min) / self.zoomFactor
        self.simulator.set_angle2_min(center2 - newHeight / 2)
        self.simulator.set_angle2_max(center2 + newHeight / 2)

        self.print_angle_boundaries()
        self.initialize_data()
        self.draw_fractal(True)


    def zoom_out(self, event):
        center1 = (event.x / self.simulator.imageResolutionPixelsWidth) * (self.simulator.angle1Max - self.simulator.angle1Min) + self.angle1Min
        newWidth = (self.simulator.angle1Max - self.simulator.angle1Min) * self.zoomFactor
        self.simulator.set_angle1_min(center1 -  newWidth/2)
        self.simulator.set_angle1_max(center1 + newWidth/2)

        center2 = (1 - event.y / self.simulator.imageResolutionPixelsHeight) * (self.simulator.angle2Max - self.simulator.angle2Min) + self.angle2Min
        newHeight = (self.simulator.angle2Max - self.simulator.angle2Min) * self.zoomFactor
        self.simulator.set_angle2_min(center2 - newHeight / 2)
        self.simulator.set_angle2_max(center2 + newHeight / 2)

        self.print_angle_boundaries()
        self.initialize_data()
        self.draw_fractal(True)


    def increase_time_to_wait_for_flip(self, event):
        self.maxTimeToSeeIfPendulumFlipsSeconds *= self.maxTimeToSeeIfPendulumFlipsChangeFactor
        self.draw_fractal(False)


    def decrease_time_to_wait_for_flip(self, event):
        self.maxTimeToSeeIfPendulumFlipsSeconds /= self.maxTimeToSeeIfPendulumFlipsChangeFactor
        self.initialize_data() # TODO create a kernel method to efficiently compute new states from the current ones
        self.draw_fractal(True)


    def decrease_time_step(self, event):
        self.simulator.timeStep /= self.timeStepFactor
        self.initialize_data()
        self.draw_fractal(True)


    def increase_time_step(self, event):
        self.simulator.timeStep *= self.timeStepFactor
        self.initialize_data()
        self.draw_fractal(True)


    def print_angle_boundaries(self):
        print('self.simulator.set_angle1_min(' + str(self.simulator.angle1Min) + ')')
        print('self.simulator.set_angle1_max(' + str(self.simulator.angle1Max) + ')')
        print('self.simulator.set_angle2_min(' + str(self.simulator.angle2Min) + ')')
        print('self.simulator.set_angle2_max(' + str(self.simulator.angle2Max) + ')')


    def initialize_data(self):
        # The data containing the pendulum states and the number of time steps before the pendulums flip.
        self.currentStates = np.zeros((4, self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))
        self.numTimeStepsTillFlipData = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(np.int32))
        self.numTimeStepsAlreadyExecuted = 0


    def draw_fractal(self, startFromDefaultState):

        print('Drawing fractal')

        # Compute the double pendulum fractal image.
        start = time.time()

        # Run the kernel.
        maxTimeStepsToExecute = self.maxTimeToSeeIfPendulumFlipsSeconds/self.simulator.timeStep
        self.simulator.compute_new_pendulum_states(self.currentStates, self.numTimeStepsTillFlipData, self.numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, startFromDefaultState)

        # Save the new pendulum states and time step till flip counts to a file so the data can be re-used in another run.
        self.numTimeStepsAlreadyExecuted = maxTimeStepsToExecute
        saveFilePath = self.directoryToSaveData + '/saved_data_for_kernel_run'
        np.savez_compressed(saveFilePath, initialStates=self.currentStates, numTimeStepsTillFlipData=self.numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted=np.array([self.numTimeStepsAlreadyExecuted]))
        logger.info('saved data to: "' + str(saveFilePath) + '"')

        # Generate an image from the current time step counts.
        redScale = 1
        greenScale = 4
        blueScale = 7.2
        shift = .11/9.81*pi
        image = self.simulator.create_image_from_num_time_steps_till_flip(self.numTimeStepsTillFlipData, redScale, greenScale, blueScale, shift)

        # Save the image so it isn't garbage collected.
        photoImage = ImageTk.PhotoImage(image)
        self.image = photoImage

        # Display the image.
        self.canvas.create_image((self.simulator.imageResolutionPixelsWidth / 2, self.simulator.imageResolutionPixelsHeight / 2), image=photoImage, state='normal')

        # Save the image to a file.
        save_image_to_file(self.directoryToSaveData, image)

        print('Total time to draw fractal = ' + str(time.time() - start))
        print('')

if __name__ == "__main__":
    app = DoublePendulumFractalApp()
    app.mainloop()

