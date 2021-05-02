import logging
import os
import random
import sys
from math import pi
from pathlib import Path

import numpy as np

from double_pendulum_kernel_methods import DoublePendulumCudaSimulator, SimulationAlgorithm, ADAPTIVE_STEP_SIZE_METHODS
from gpu.utils import save_image_to_file

logger = logging.getLogger('root')

class GenerateDoublePendulumFractalImages:

    # Configuration
    deviceNumberToUse = 0
    useDoublePrecision = False
    # algorithm = SimulationAlgorithm.RK_4
    # algorithm = SimulationAlgorithm.RKF_45
    # algorithm = SimulationAlgorithm.CASH_KARP_45
    algorithm = SimulationAlgorithm.DORMAND_PRINCE_54
    
    # Used to color the image based on how long it took a pendulum to flip.
    redScale = 1
    greenScale = 4
    blueScale = 7.2
    shift = .11/9.81*pi

    def __init__(self, directoryToSaveData):
        # The directory used to store the image and pendulum data files.
        self.directoryToSaveData = directoryToSaveData
        Path(directoryToSaveData).mkdir(parents=True, exist_ok=True)

        # Initialize the logger.
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(self.directoryToSaveData + '/log.log'))

        # Initialize the simulator.
        self.initialize_simulator()


    def initialize_simulator(self):

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
        self.simulator.set_image_width_pixels(int(1000/2**1))

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


    def generate_images_from_scratch(self, numImagesToCreate, maxTimeToExecuteInTotal, simulationTimeBetweenSaves):
        initialStates = np.zeros((4, self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))
        firstImageComputed = False

        if self.algorithm is SimulationAlgorithm.RK_4:
            numTimeStepsTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(np.int32))
            maxTimeStepsToExecuteInTotal = maxTimeToExecuteInTotal/self.simulator.timeStep
            self.generate_images_rk4(self.directoryToSaveData, numImagesToCreate, initialStates, numTimeStepsTillFlip, 0, maxTimeStepsToExecuteInTotal, firstImageComputed, simulationTimeBetweenSaves)

        elif self.algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            timeTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), self.simulator.npFloatType)
            self.generate_images_adaptive_time_step(self.directoryToSaveData, numImagesToCreate, initialStates, timeTillFlip, 0, maxTimeToExecuteInTotal, firstImageComputed, simulationTimeBetweenSaves)


    def generate_images_from_save(self, numImagesToCreate, saveFile, simulationTimeBetweenSaves):
        directory = os.path.dirname(saveFile)
        loaded = np.load(saveFile)
        initialStates = loaded['initialStates']
        firstImageComputed = True

        if self.algorithm is SimulationAlgorithm.RK_4:
            numTimeStepsTillFlipData = loaded['numTimeStepsTillFlipData']
            numTimeStepsAlreadyExecuted = loaded['numTimeStepsAlreadyExecuted'][0]
            maxTimeStepsToExecute = 2*numTimeStepsAlreadyExecuted
            self.generate_images_rk4(directory, numImagesToCreate, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed, simulationTimeBetweenSaves)

        elif self.algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            timeTillFlipData = loaded['timeTillFlipData']
            timeAlreadyExecuted = loaded['timeAlreadyExecuted'][0]
            maxTimeToExecute = 2*timeAlreadyExecuted
            self.generate_images_adaptive_time_step(directory, numImagesToCreate, initialStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, firstImageComputed, simulationTimeBetweenSaves)


    def generate_images_rk4(self, directory, numImagesToCreate, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed, simulationTimeBetweenSaves):
        for i in range(0, numImagesToCreate):
            saveFilePath = directory + '/saved_data_for_kernel_run'
            self.generate_image_rk4(saveFilePath, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed, simulationTimeBetweenSaves)
            logger.info('')

            numTimeStepsAlreadyExecuted = maxTimeStepsToExecute
            maxTimeStepsToExecute *= 2
            firstImageComputed = True


    def generate_images_adaptive_time_step(self, directory, numImagesToCreate, initialStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, firstImageComputed, simulationTimeBetweenSaves):
        for i in range(0, numImagesToCreate):
            saveFilePath = directory + '/saved_data_for_kernel_run'
            self.generate_image_adaptive_step_size_method(saveFilePath, initialStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, firstImageComputed, simulationTimeBetweenSaves)
            logger.info('')

            timeAlreadyExecuted = maxTimeToExecute
            maxTimeToExecute *= 2
            firstImageComputed = True


    def generate_image_rk4(self, saveFilePath, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed, simulationTimeBetweenSaves):
        # Run the kernel the given amount of simulation time between saves, saving after each run.
        simulationTimeStepsToExecuteForImage = maxTimeStepsToExecute - numTimeStepsAlreadyExecuted
        simulationTimeStepsBetweenSaves = min(simulationTimeStepsToExecuteForImage, simulationTimeBetweenSaves/self.simulator.timeStep)
        for i in range(int(simulationTimeStepsToExecuteForImage/simulationTimeStepsBetweenSaves)):
            curMaxTimeStepsToExecute = numTimeStepsAlreadyExecuted + simulationTimeStepsBetweenSaves
            self.simulator.compute_new_pendulum_states_rk4(initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, curMaxTimeStepsToExecute, not firstImageComputed)
            numTimeStepsAlreadyExecuted = curMaxTimeStepsToExecute

            # Save the new pendulum states and time step till flip counts to a file so the data can be re-used in another run.
            np.savez_compressed(saveFilePath, initialStates=initialStates, numTimeStepsTillFlipData=numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted=np.array([curMaxTimeStepsToExecute]))
            logger.info('saved data to: "' + str(saveFilePath) + '"')

        # Generate an image from the current time step counts.
        image = self.simulator.create_image_from_number_of_time_steps_till_flip(numTimeStepsTillFlipData, self.redScale, self.greenScale, self.blueScale, self.shift)

        # Save the image to a file.
        save_image_to_file(self.directoryToSaveData, image)

        return image


    def generate_image_adaptive_step_size_method(self, saveFilePath, initialStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, firstImageComputed, simulationTimeBetweenSaves):
        # Run the kernel the given amount of simulation time between saves, saving after each run.
        simulationTimeToExecuteForImage = maxTimeToExecute - timeAlreadyExecuted
        simulationTimeBetweenSaves = min(simulationTimeBetweenSaves, simulationTimeToExecuteForImage)
        for i in range(int(simulationTimeToExecuteForImage/simulationTimeBetweenSaves)):
            curMaxTimeToExecute = timeAlreadyExecuted + simulationTimeBetweenSaves
            self.simulator.compute_new_pendulum_states_runge_kutta_adaptive_step_size(initialStates, timeTillFlipData, timeAlreadyExecuted, curMaxTimeToExecute, not firstImageComputed)
            timeAlreadyExecuted = curMaxTimeToExecute

            # Save the new pendulum states and time step till flip counts to a file so the data can be re-used in another run.
            np.savez_compressed(saveFilePath, initialStates=initialStates, timeTillFlipData=timeTillFlipData, timeAlreadyExecuted=np.array([curMaxTimeToExecute]))
            logger.info('saved data to: "' + str(saveFilePath) + '"')

        # Generate an image from the current time step counts.
        image = self.simulator.create_image_from_time_till_flip(timeTillFlipData, self.redScale, self.greenScale, self.blueScale, self.shift)

        # Save the image to a file.
        save_image_to_file(self.directoryToSaveData, image)

        return image


    def generate_random_color_images(self, numImagesToCreate, maxTimeToSeeIfPendulumFlipsSeconds):
        # Only the Runge-Kutta-Felhberg method is supported.
        self.algorithm = SimulationAlgorithm.CASH_KARP_45
        self.initialize_simulator()

        # Generate the images.
        for i in range(0, numImagesToCreate):
            # Run the kernel.
            initialStates = np.zeros((4, self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))
            timeTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), self.simulator.npFloatType)
            self.simulator.compute_new_pendulum_states_runge_kutta_adaptive_step_size(initialStates, timeTillFlip, 0, maxTimeToSeeIfPendulumFlipsSeconds, True)

            # Generate an image from the current time step counts, using random parameters for the coloring.
            redScale = random.uniform(0, 10)
            greenScale = random.uniform(0, 10)
            blueScale = random.uniform(0, 10)
            shift = random.uniform(0, 1)
            image = self.simulator.create_image_from_time_till_flip(timeTillFlip, redScale, greenScale, blueScale, shift)

            # Save the image to a file.
            save_image_to_file(self.directoryToSaveData, image)
            logger.info('')


if __name__ == "__main__":
    # app = GenerateDoublePendulumFractalImages(create_directory())
    app = GenerateDoublePendulumFractalImages('./tmp')

    # Run the program to generate double pendulum fractal images.
    app.generate_images_from_scratch(3, 2**6, 2**6)
    # app.generate_random_color_images(10, 2**4)
    # app.generate_images_from_save(1, './tmp/saved_data_for_kernel_run.npz', 2**8)


