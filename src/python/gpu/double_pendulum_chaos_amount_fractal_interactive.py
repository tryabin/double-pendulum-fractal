import logging
import sys
import time
import tkinter as tk
from math import *
from pathlib import Path

import numpy as np
from PIL import ImageTk
from win32api import GetSystemMetrics

from double_pendulum_kernel_methods import DoublePendulumCudaSimulator, SimulationAlgorithm
from gpu.double_pendulum_fractal_generate_images import GenerateDoublePendulumFractalImages
from utils import save_image_to_file

logger = logging.getLogger(__name__)

class DoublePendulumChaosAmountFractalApp(tk.Tk):
    # UI control parameters
    zoomFactor = 2
    maxTimeToSimulateFactor = 2
    timeStepFactor = 2
    errorToleranceFactor = 2
    maxTimeToSimulateSeconds = 2**5
    differenceCutoff = 1e-0

    # Used for optimization so pendulums that have become chaotic are not simulated.
    simulationTimeBetweenComputingChaosAmount = 2**3

    # Other parameters
    deviceNumberToUse = 0  # The GPU to use to run the simulation.
    useDoublePrecision = False # The type of floating point arithmetic to use in the simulation.
    # algorithm = SimulationAlgorithm.RKF_45
    # algorithm = SimulationAlgorithm.CASH_KARP_45
    algorithm = SimulationAlgorithm.DORMAND_PRINCE_54
    # algorithm = SimulationAlgorithm.FEHLBERG_87

    def __init__(self):
        tk.Tk.__init__(self)

        # Initialize the logger.
        self.directoryToSaveData = './interactive chaos amount'
        Path(self.directoryToSaveData).mkdir(parents=True, exist_ok=True)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(self.directoryToSaveData + '/log.log', mode='w'))

        # Initialize the simulator.
        self.simulator = DoublePendulumCudaSimulator(self.deviceNumberToUse, self.directoryToSaveData, self.useDoublePrecision, self.algorithm, None)

        # The range of pendulum angles.

        # High energy area fully zoomed out
        # self.simulator.set_angle1_min(-2*pi)
        # self.simulator.set_angle1_max(0)
        # self.simulator.set_angle2_min(0)
        # self.simulator.set_angle2_max(2*pi)

        # High energy area zoomed in 2x
        self.simulator.set_angle1_min(-2*pi + 2*pi/4)
        self.simulator.set_angle1_max(0 - 2*pi/4)
        self.simulator.set_angle2_min(0 + 2*pi/4)
        self.simulator.set_angle2_max(2*pi - 2*pi/4)

        # self.simulator.set_angle1_min(-3.396454357612266)
        # self.simulator.set_angle1_max(-3.371910665006095)
        # self.simulator.set_angle2_min(1.901448953585222)
        # self.simulator.set_angle2_max(1.925992646191392)

        # self.simulator.set_angle1_min(pi/2)
        # self.simulator.set_angle1_max(2*pi)
        # self.simulator.set_angle2_min(pi/2)
        # self.simulator.set_angle2_max(2*pi)

        # self.simulator.set_angle1_min(-3*pi)
        # self.simulator.set_angle1_max(-pi)
        # self.simulator.set_angle2_min(-pi)
        # self.simulator.set_angle2_max(pi)

        # The width of the image in pixels.
        self.simulator.set_image_dimensions_based_on_width(1000/2**0)

        # The amount of super-sampling anti-aliasing to apply to the image. Can be fractional.
        # 1 means no anti-aliasing.
        # 2 means four total samples are used per pixel.
        # 3 means nine total samples are used per pixel, etc.
        self.simulator.set_anti_aliasing_amount(1)

        # Simulation parameters.
        self.simulator.set_time_step(.01/2**2)
        self.simulator.set_error_tolerance(1e-6)
        self.simulator.set_gravity(1)
        self.simulator.set_point1_mass(1)
        self.simulator.set_point2_mass(1)
        self.simulator.set_pendulum1_length(1)
        self.simulator.set_pendulum2_length(1)

        # Initialize the object used to generate images.
        self.imageGenerator = GenerateDoublePendulumFractalImages(self.simulator)

        # Initialize the data containing the pendulum states and the number of time steps before the pendulums flip.
        self.initialize_data()

        # Set the window size and position.
        self.geometry('%dx%d+%d+%d'%(self.simulator.imageResolutionWidthPixels,
                                     self.simulator.imageResolutionHeightPixels,
                                     GetSystemMetrics(0)/2 - self.simulator.imageResolutionWidthPixels/2,
                                     GetSystemMetrics(1)/2 - self.simulator.imageResolutionHeightPixels/2))

        self.canvas = tk.Canvas(self, width=self.simulator.imageResolutionWidthPixels, height=self.simulator.imageResolutionHeightPixels)
        self.canvas.pack(side='top', fill='both', expand=True)
        self.canvas.bind('z', self.zoom_in)
        self.canvas.bind('x', self.zoom_out)
        self.canvas.bind('a', self.increase_time_to_wait_for_flip)
        self.canvas.bind('s', self.decrease_time_to_wait_for_flip)
        self.canvas.bind('q', self.increase_accuracy)
        self.canvas.bind('w', self.decrease_accuracy)
        self.canvas.focus_set()
        
        self.draw_fractal()


    def zoom_in(self, event):
        center1 = (event.x/self.simulator.imageResolutionWidthPixels)*(self.simulator.angle1Max - self.simulator.angle1Min) + self.simulator.angle1Min
        newWidth = (self.simulator.angle1Max - self.simulator.angle1Min) / self.zoomFactor
        self.simulator.set_angle1_min(center1 -  newWidth/2)
        self.simulator.set_angle1_max(center1 + newWidth/2)

        center2 = (1 - event.y/self.simulator.imageResolutionHeightPixels)*(self.simulator.angle2Max - self.simulator.angle2Min) + self.simulator.angle2Min
        newHeight = (self.simulator.angle2Max - self.simulator.angle2Min) / self.zoomFactor
        self.simulator.set_angle2_min(center2 - newHeight / 2)
        self.simulator.set_angle2_max(center2 + newHeight / 2)

        self.print_angle_boundaries()
        self.initialize_data()
        self.draw_fractal()


    def zoom_out(self, event):
        center1 = (event.x/self.simulator.imageResolutionWidthPixels)*(self.simulator.angle1Max - self.simulator.angle1Min) + self.simulator.angle1Min
        newWidth = (self.simulator.angle1Max - self.simulator.angle1Min) * self.zoomFactor
        self.simulator.set_angle1_min(center1 -  newWidth/2)
        self.simulator.set_angle1_max(center1 + newWidth/2)

        center2 = (1 - event.y/self.simulator.imageResolutionHeightPixels)*(self.simulator.angle2Max - self.simulator.angle2Min) + self.simulator.angle2Min
        newHeight = (self.simulator.angle2Max - self.simulator.angle2Min) * self.zoomFactor
        self.simulator.set_angle2_min(center2 - newHeight / 2)
        self.simulator.set_angle2_max(center2 + newHeight / 2)

        self.print_angle_boundaries()
        self.initialize_data()
        self.draw_fractal()


    def increase_time_to_wait_for_flip(self, event):
        self.maxTimeToSimulateSeconds *= self.maxTimeToSimulateFactor
        self.draw_fractal()


    def decrease_time_to_wait_for_flip(self, event):
        self.maxTimeToSimulateSeconds /= self.maxTimeToSimulateFactor
        self.initialize_data() # TODO create a kernel method to efficiently compute new states from the current ones
        self.draw_fractal()


    def increase_accuracy(self, event):
        if self.algorithm is SimulationAlgorithm.RK_4:
            self.simulator.timeStep /= self.timeStepFactor
        else:
            self.simulator.errorTolerance /= self.errorToleranceFactor
        self.initialize_data()
        self.draw_fractal()


    def decrease_accuracy(self, event):
        if self.algorithm is SimulationAlgorithm.RK_4:
            self.simulator.timeStep *= self.timeStepFactor
        else:
            self.simulator.errorTolerance *= self.errorToleranceFactor
        self.initialize_data()
        self.draw_fractal()


    def print_angle_boundaries(self):
        logger.info('self.simulator.set_angle1_min(' + str(self.simulator.angle1Min) + ')')
        logger.info('self.simulator.set_angle1_max(' + str(self.simulator.angle1Max) + ')')
        logger.info('self.simulator.set_angle2_min(' + str(self.simulator.angle2Min) + ')')
        logger.info('self.simulator.set_angle2_max(' + str(self.simulator.angle2Max) + ')')


    def initialize_data(self):
        self.timeAlreadySimulated = 0

        # The array containing the pendulum states.
        self.currentStates = np.zeros((4, self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))

        # The array containing the amount of chaos for each pendulum.
        # -1 indicates the amount of chaos has not been calculated yet.
        self.chaosAmountData = -1*np.ones((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))


    def draw_fractal(self):
        logger.info('Drawing fractal')

        # Compute the double pendulum fractal image.
        start = time.time()

        # Generate the image
        computedImage = self.imageGenerator.generate_chaos_amount_image(self.currentStates,
                                                                        self.chaosAmountData,
                                                                        self.maxTimeToSimulateSeconds,
                                                                        self.simulationTimeBetweenComputingChaosAmount,
                                                                        self.differenceCutoff,
                                                                        self.timeAlreadySimulated)

        # Save the new pendulum states and time step till flip counts to a file so the data can be re-used in another run.
        saveFilePath = self.directoryToSaveData + '/saved_data_for_kernel_run'
        self.timeAlreadySimulated = self.maxTimeToSimulateSeconds
        np.savez_compressed(saveFilePath, initialStates=self.currentStates, amountOfTimeAlreadyExecuted=np.array([self.timeAlreadySimulated]))
        logger.info('saved data to: "' + str(saveFilePath) + '"')

        # Save the image to a file.
        save_image_to_file(self.directoryToSaveData, computedImage)

        # Save the rendered image so it isn't garbage collected.
        self.renderedImage = ImageTk.PhotoImage(computedImage)

        # Display the image.
        self.canvas.create_image((self.simulator.imageResolutionWidthPixels/2, self.simulator.imageResolutionHeightPixels/2), image=self.renderedImage, state='normal')

        logger.info('Total time to draw fractal = ' + str(time.time() - start))
        logger.info('')

if __name__ == "__main__":
    app = DoublePendulumChaosAmountFractalApp()
    app.mainloop()

