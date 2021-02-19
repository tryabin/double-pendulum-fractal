import logging
import os
import random
import sys
from math import pi

import numpy as np

from double_pendulum_kernel_methods import DoublePendulumCudaSimulator
from utils import create_directory, save_image_to_file

logger = logging.getLogger('root')

class GenerateDoublePendulumFractalImages:

    def __init__(self, directoryToSaveData):
        # The directory used to store the image and pendulum data files.
        self.directoryToSaveData = directoryToSaveData

        # Initialize the logger.
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(self.directoryToSaveData + '/log.log'))

        # Initialize the simulator.
        deviceNumberToUse = 0
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
        self.simulator.set_image_width_pixels(int(1000/2**0))

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


    def generate_images_from_scratch(self, numImagesToCreate, initialTimeToExecute):
        initialStates = np.zeros((4, self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))
        numTimeStepsTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(np.int32))
        maxTimeStepsToExecuteInTotal = initialTimeToExecute / self.simulator.timeStep
        firstImageComputed = False
        self.generate_images(self.directoryToSaveData, numImagesToCreate, initialStates, numTimeStepsTillFlip, maxTimeStepsToExecuteInTotal, 0, firstImageComputed)


    def generate_images_from_save(self, numImagesToCreate, saveFile):
        directory = os.path.dirname(saveFile)
        loaded = np.load(saveFile)
        initialStates = loaded['initialStates']
        numTimeStepsTillFlipData = loaded['numTimeStepsTillFlipData']
        numTimeStepsAlreadyExecuted = loaded['numTimeStepsAlreadyExecuted'][0]
        maxTimeStepsToExecute = 2*numTimeStepsAlreadyExecuted
        firstImageComputed =  True
        self.generate_images(directory, numImagesToCreate, initialStates, numTimeStepsTillFlipData, maxTimeStepsToExecute, numTimeStepsAlreadyExecuted, firstImageComputed)


    def generate_images(self, directory, numImagesToCreate, initialStates, numTimeStepsTillFlipData, maxTimeStepsToExecute, numTimeStepsAlreadyExecuted, firstImageComputed):
        # Generate the images.
        for i in range(0, numImagesToCreate):
            saveFilePath = directory + '/saved_data_for_kernel_run_' + str(i)
            self.generate_image(saveFilePath, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed)
            logger.info('')

            numTimeStepsAlreadyExecuted = maxTimeStepsToExecute
            maxTimeStepsToExecute *= 2
            firstImageComputed = True


    def generate_image(self, saveFilePath, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed):
        # Run the kernel.
        self.simulator.compute_new_pendulum_states(initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, not firstImageComputed)

        # Save the new pendulum states and time step till flip counts to a file so the data can be re-used in another run.
        np.savez_compressed(saveFilePath, initialStates=initialStates, numTimeStepsTillFlipData=numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted=np.array([maxTimeStepsToExecute]))
        logger.info('saved data to: "' + str(saveFilePath) + '"')

        # Generate an image from the current time step counts.
        redScale = 1
        greenScale = 4
        blueScale = 7.2
        shift = .11 / 9.81 * pi
        image = self.simulator.create_image_from_num_time_steps_till_flip(numTimeStepsTillFlipData, redScale, greenScale, blueScale, shift)

        # Save the image to a file.
        save_image_to_file(self.directoryToSaveData, image)

        return image


    def generate_random_color_images(self, numImagesToCreate, maxTimeToSeeIfPendulumFlipsSeconds):

        maxNumberOfTimeStepsToSeeIfPendulumFlips = maxTimeToSeeIfPendulumFlipsSeconds / self.simulator.timeStep

        # Generate the images.
        for i in range(0, numImagesToCreate):
            # Run the kernel.
            initialStates = np.zeros((4, self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))
            numTimeStepsTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(np.int32))
            self.simulator.compute_new_pendulum_states(initialStates, numTimeStepsTillFlip, 0, maxNumberOfTimeStepsToSeeIfPendulumFlips, True)

            # Generate an image from the current time step counts, using random parameters for the coloring.
            redScale = random.uniform(0, 10)
            greenScale = random.uniform(0, 10)
            blueScale = random.uniform(0, 10)
            shift = random.uniform(0, 1)
            image = self.simulator.create_image_from_num_time_steps_till_flip(numTimeStepsTillFlip, redScale, greenScale, blueScale, shift)

            # Save the image to a file.
            save_image_to_file(self.directoryToSaveData, image)
            logger.info('')


if __name__ == "__main__":
    # app = GenerateDoublePendulumFractalImages(create_directory())
    app = GenerateDoublePendulumFractalImages('./tmp')

    # Run the program to generate double pendulum fractal images.
    app.generate_images_from_scratch(1, 2**6)
    # app.generate_random_color_images(10, 2**4)
    # app.generate_images_from_save(1, './tmp/saved_data_for_kernel_run_0.npz')




