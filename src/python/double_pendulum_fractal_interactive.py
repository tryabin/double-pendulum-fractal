import logging
import sys
import time
import tkinter as tk
from math import *
from pathlib import Path

import numpy as np
from PIL import ImageTk
from win32api import GetSystemMetrics

from double_pendulum_kernel_methods import DoublePendulumCudaSimulator, SimulationAlgorithm, ADAPTIVE_STEP_SIZE_METHODS
from utils import save_image_to_file

logger = logging.getLogger('root')

class DoublePendulumFractalApp(tk.Tk):
    # UI control parameters.
    zoomFactor = 2
    maxTimeToSeeIfPendulumFlipsChangeFactor = 2
    timeStepFactor = 2
    errorToleranceFactor = 2
    maxTimeToSeeIfPendulumFlipsSeconds = 2**6

    # Other parameters.
    deviceNumberToUse = 0  # The GPU to use to run the simulation.
    useDoublePrecision = False # The type of floating point arithmetic to use in the simulation.
    # algorithm = SimulationAlgorithm.RK4
    # algorithm = SimulationAlgorithm.RKF45
    algorithm = SimulationAlgorithm.CASH_KARP

    def __init__(self):
        tk.Tk.__init__(self)

        # Initialize the logger.
        self.directoryToSaveData = './interactive'
        Path(self.directoryToSaveData).mkdir(parents=True, exist_ok=True)

        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(self.directoryToSaveData + '/log.log', mode='w'))

        # Initialize the simulator.
        self.simulator = DoublePendulumCudaSimulator(self.deviceNumberToUse, self.directoryToSaveData, self.useDoublePrecision, self.algorithm)

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
        self.simulator.set_error_tolerance(1e-7)
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
        self.canvas.bind('q', self.increase_accuracy)
        self.canvas.bind('w', self.decrease_accuracy)
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
        center1 = (event.x / self.simulator.imageResolutionPixelsWidth) * (self.simulator.angle1Max - self.simulator.angle1Min) + self.simulator.angle1Min
        newWidth = (self.simulator.angle1Max - self.simulator.angle1Min) * self.zoomFactor
        self.simulator.set_angle1_min(center1 -  newWidth/2)
        self.simulator.set_angle1_max(center1 + newWidth/2)

        center2 = (1 - event.y / self.simulator.imageResolutionPixelsHeight) * (self.simulator.angle2Max - self.simulator.angle2Min) + self.simulator.angle2Min
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


    def increase_accuracy(self, event):
        if self.algorithm is SimulationAlgorithm.RK4:
            self.simulator.timeStep /= self.timeStepFactor
        else:
            self.simulator.errorTolerance /= self.errorToleranceFactor
        self.initialize_data()
        self.draw_fractal(True)


    def decrease_accuracy(self, event):
        if self.algorithm is SimulationAlgorithm.RK4:
            self.simulator.timeStep *= self.timeStepFactor
        else:
            self.simulator.errorTolerance *= self.errorToleranceFactor
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

        # The data used by the RK4 method.
        self.numTimeStepsTillFlipData = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(np.int32))
        self.numTimeStepsAlreadyExecuted = 0

        # The data used by the RKF45 method.
        npFloatType = np.float64 if self.useDoublePrecision else np.float32
        self.timeTillFlipData = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), npFloatType)
        self.amountOfTimeAlreadyExecuted = 0


    def draw_fractal(self, startFromDefaultState):

        print('Drawing fractal')

        # Compute the double pendulum fractal image.
        start = time.time()

        # Run the kernel.
        if self.algorithm is SimulationAlgorithm.RK4:
            self.simulator.compute_new_pendulum_states_rk4(self.currentStates, self.numTimeStepsTillFlipData, self.numTimeStepsAlreadyExecuted, self.maxTimeToSeeIfPendulumFlipsSeconds/self.simulator.timeStep, startFromDefaultState)
        elif self.algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            self.simulator.compute_new_pendulum_states_runge_kutta_adaptive_step_size(self.currentStates, self.timeTillFlipData, self.amountOfTimeAlreadyExecuted, self.maxTimeToSeeIfPendulumFlipsSeconds, startFromDefaultState)

        # Save the new pendulum states and time step till flip counts to a file so the data can be re-used in another run.
        saveFilePath = self.directoryToSaveData + '/saved_data_for_kernel_run'
        if self.algorithm is SimulationAlgorithm.RK4:
            self.numTimeStepsAlreadyExecuted = self.maxTimeToSeeIfPendulumFlipsSeconds/self.simulator.timeStep
            np.savez_compressed(saveFilePath, initialStates=self.currentStates, numTimeStepsTillFlipData=self.numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted=np.array([self.numTimeStepsAlreadyExecuted]))
        elif self.algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            self.amountOfTimeAlreadyExecuted = self.maxTimeToSeeIfPendulumFlipsSeconds
            np.savez_compressed(saveFilePath, initialStates=self.currentStates, timeTillFlipData=self.timeTillFlipData, amountOfTimeAlreadyExecuted=np.array([self.amountOfTimeAlreadyExecuted]))
        logger.info('saved data to: "' + str(saveFilePath) + '"')

        # Generate an image from the current time step counts.
        redScale = 1
        greenScale = 4
        blueScale = 7.2
        shift = .11/9.81*pi
        computedImage = None
        if self.algorithm is SimulationAlgorithm.RK4:
            computedImage = self.simulator.create_image_from_number_of_time_steps_till_flip(self.numTimeStepsTillFlipData, redScale, greenScale, blueScale, shift)
        elif self.algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            computedImage = self.simulator.create_image_from_time_till_flip(self.timeTillFlipData, redScale, greenScale, blueScale, shift)

        # Save the image to a file.
        save_image_to_file(self.directoryToSaveData, computedImage)

        # Save the rendered image so it isn't garbage collected.
        self.renderedImage = ImageTk.PhotoImage(computedImage)

        # Display the image.
        self.canvas.create_image((self.simulator.imageResolutionPixelsWidth / 2, self.simulator.imageResolutionPixelsHeight / 2), image=self.renderedImage, state='normal')

        print('Total time to draw fractal = ' + str(time.time() - start))
        print('')

if __name__ == "__main__":
    app = DoublePendulumFractalApp()
    app.mainloop()

