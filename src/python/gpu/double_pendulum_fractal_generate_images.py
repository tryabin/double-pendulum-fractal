import logging
import os
import random
import sys
import time
from math import pi, log, exp, inf
from pathlib import Path

import numpy as np

from double_pendulum_kernel_methods import DoublePendulumCudaSimulator, SimulationAlgorithm, ADAPTIVE_STEP_SIZE_METHODS
from gpu.utils import save_image_to_file, create_directory, create_video_from_images

logger = logging.getLogger(__name__)

class GenerateDoublePendulumFractalImages:

    # Used to color the image based on how long it took a pendulum to flip.
    redScale = 1
    greenScale = 4
    blueScale = 7.2
    shift = .11/9.81*pi

    def __init__(self, simulator : DoublePendulumCudaSimulator):

        # The object used to run the CUDA kernels.
        self.simulator = simulator
        
        # The directory used to store the image and pendulum data files.
        self.directoryToSaveData = simulator.get_directory_to_save_data()
        Path(self.directoryToSaveData).mkdir(parents=True, exist_ok=True)

        # Initialize the logger.
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(self.directoryToSaveData + '/log.log'))


    def generate_images_from_scratch(self, numImagesToCreate, maxTimeToExecuteInTotal, simulationTimeBetweenSaves):
        initialStates = np.zeros((4, self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))
        firstImageComputed = False

        if simulator.algorithm is SimulationAlgorithm.RK_4:
            numTimeStepsTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(np.int32))
            maxTimeStepsToExecuteInTotal = maxTimeToExecuteInTotal/self.simulator.timeStep
            self.generate_images_time_till_flip_rk4(self.directoryToSaveData, numImagesToCreate, initialStates, numTimeStepsTillFlip, 0, maxTimeStepsToExecuteInTotal, firstImageComputed, simulationTimeBetweenSaves)

        elif simulator.algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            timeTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), self.simulator.npFloatType)
            self.generate_images_time_till_flip_adaptive_time_step(self.directoryToSaveData, numImagesToCreate, initialStates, timeTillFlip, 0, maxTimeToExecuteInTotal, firstImageComputed, simulationTimeBetweenSaves)


    def generate_images_from_save_time_till_flip_method(self, numImagesToCreate, saveFile, simulationTimeBetweenSaves):
        directory = os.path.dirname(saveFile)
        savedData = np.load(saveFile)
        initialStates = savedData['initialStates']
        firstImageComputed = True

        if simulator.algorithm is SimulationAlgorithm.RK_4:
            numTimeStepsTillFlipData = savedData['numTimeStepsTillFlipData']
            numTimeStepsAlreadyExecuted = savedData['numTimeStepsAlreadyExecuted'][0]
            maxTimeStepsToExecute = 2*numTimeStepsAlreadyExecuted
            self.generate_images_time_till_flip_rk4(directory, numImagesToCreate, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed, simulationTimeBetweenSaves)

        elif simulator.algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            timeTillFlipData = savedData['timeTillFlipData']
            timeAlreadyExecuted = savedData['timeAlreadyExecuted'][0]
            maxTimeToExecute = 2*timeAlreadyExecuted
            self.generate_images_time_till_flip_adaptive_time_step(directory, numImagesToCreate, initialStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, firstImageComputed, simulationTimeBetweenSaves)


    def generate_images_time_till_flip_rk4(self, directory, numImagesToCreate, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed, simulationTimeBetweenSaves):
        for i in range(0, numImagesToCreate):
            saveFilePath = directory + '/saved_data_for_kernel_run'
            self.generate_image_time_till_flip_rk4(saveFilePath, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed, simulationTimeBetweenSaves)
            logger.info('')

            numTimeStepsAlreadyExecuted = maxTimeStepsToExecute
            maxTimeStepsToExecute *= 2
            firstImageComputed = True


    def generate_images_time_till_flip_adaptive_time_step(self, directory, numImagesToCreate, initialStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, firstImageComputed, simulationTimeBetweenSaves):
        for i in range(0, numImagesToCreate):
            saveFilePath = directory + '/saved_data_for_kernel_run'
            self.generate_image_time_till_flip_adaptive_step_size_method(saveFilePath, initialStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, firstImageComputed, simulationTimeBetweenSaves)
            logger.info('')

            timeAlreadyExecuted = maxTimeToExecute
            maxTimeToExecute *= 2
            firstImageComputed = True


    def generate_image_time_till_flip_rk4(self, saveFilePath, initialStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, firstImageComputed, simulationTimeBetweenSaves):
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


    def generate_image_time_till_flip_adaptive_step_size_method(self, saveFilePath, initialStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, firstImageComputed, simulationTimeBetweenSaves):
        # Run the kernel the given amount of simulation time between saves, saving after each run.
        simulationTimeToExecuteForImage = maxTimeToExecute - timeAlreadyExecuted
        simulationTimeBetweenSaves = min(simulationTimeBetweenSaves, simulationTimeToExecuteForImage)
        for i in range(int(simulationTimeToExecuteForImage/simulationTimeBetweenSaves)):
            curMaxTimeToExecute = timeAlreadyExecuted + simulationTimeBetweenSaves
            self.simulator.compute_new_pendulum_states_time_till_flip_adaptive_step_size_method(initialStates, timeTillFlipData, timeAlreadyExecuted, curMaxTimeToExecute, not firstImageComputed)
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
        # Generate the images.
        for i in range(0, numImagesToCreate):
            # Run the kernel.
            initialStates = np.zeros((4, self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))
            timeTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), self.simulator.npFloatType)
            self.simulator.compute_new_pendulum_states_time_till_flip_adaptive_step_size_method(initialStates, timeTillFlip, 0, maxTimeToSeeIfPendulumFlipsSeconds, True)

            # Generate an image from the current time step counts, using random parameters for the coloring.
            redScale = random.uniform(0, 10)
            greenScale = random.uniform(0, 10)
            blueScale = random.uniform(0, 10)
            shift = random.uniform(0, 1)
            image = self.simulator.create_image_from_time_till_flip(timeTillFlip, redScale, greenScale, blueScale, shift)

            # Save the image to a file.
            save_image_to_file(self.directoryToSaveData, image)
            logger.info('')


    def generate_chaos_amount_image_from_scratch(self, totalSimulationTime, simulationTimeBetweenComputingChaosAmount, differenceCutoff):
        # Initialize data structures
        currentStates = np.zeros((4, app.simulator.numberOfAnglesToTestY, app.simulator.numberOfAnglesToTestX), np.dtype(app.simulator.npFloatType))
        chaosAmountData = -1*np.ones((app.simulator.numberOfAnglesToTestY, app.simulator.numberOfAnglesToTestX), np.dtype(app.simulator.npFloatType))
        return self.generate_chaos_amount_image(currentStates, chaosAmountData, totalSimulationTime, simulationTimeBetweenComputingChaosAmount, differenceCutoff)


    def generate_chaos_amount_image(self, currentStates, chaosAmountData, totalSimulationTime, simulationTimeBetweenComputingChaosAmount, differenceCutoff, timeAlreadySimulated=0):
        # Run the simulation
        improvementCutoff = 1.1
        minKernelTimeSeconds = 5
        firstIterationRuntime = inf
        previousSimulationRuntime = inf
        simulateTillEnd = False
        startTimeToGenerateImage = time.time()
        while timeAlreadySimulated < totalSimulationTime:
            # Compute the amount of time to simulate on this loop. If there will be negligible gain from
            # computing the amount of chaos in between then just simulate till the end.
            curTimeToSimulate = min(simulationTimeBetweenComputingChaosAmount, totalSimulationTime - timeAlreadySimulated)
            if simulateTillEnd:
                curTimeToSimulate = totalSimulationTime - timeAlreadySimulated

            # Run the simulation for this iteration
            start = time.time()
            self.simulator.compute_new_pendulum_states_amount_of_chaos_adaptive_step_size_method(currentStates, chaosAmountData, timeAlreadySimulated, timeAlreadySimulated + curTimeToSimulate, timeAlreadySimulated == 0)
            curSimulationRuntime = time.time() - start

            # Compute the chaos amount
            self.simulator.compute_chaos_amount_from_pendulum_states(currentStates, chaosAmountData, differenceCutoff)
            timeAlreadySimulated += curTimeToSimulate

            # See if there was a negligible performance gain from the last run.
            runtimeRatioFromPreviousRun = previousSimulationRuntime/curSimulationRuntime
            runtimeRatioFromFirstRun = firstIterationRuntime/curSimulationRuntime
            if runtimeRatioFromPreviousRun < improvementCutoff < runtimeRatioFromFirstRun:
                simulateTillEnd = True

            # Store the runtimes
            if firstIterationRuntime == inf:
                firstIterationRuntime = curSimulationRuntime
            previousSimulationRuntime = curSimulationRuntime

            # Increase the time to simulate to meet the minimum runtime threshold.
            if curSimulationRuntime < minKernelTimeSeconds:
                simulationTimeBetweenComputingChaosAmount = minKernelTimeSeconds/curSimulationRuntime*curTimeToSimulate

        image = self.simulator.create_image_from_amount_of_chaos(currentStates, chaosAmountData, differenceCutoff)
        logger.info('Image generated in ' + str(time.time() - startTimeToGenerateImage) + ' seconds')
        return image


    def generate_zoom_in_video_chaos_amount_method(self):
        # Setup
        differenceCutoff = 1e-0
        fps = 5
        totalZoomTimeSeconds = 4
        totalNumberOfImages = fps*totalZoomTimeSeconds
        initialBoxWidth = pi
        initialMaxTimeToSimulate = 2**3
        finalBoxWidth = 2e-5
        boxCenterX = 3.3537026965490882
        boxCenterY = 3.2536503336400364
        finalMaxTimeToSimulate = 2**8 + 2**6
        zoomFactor = exp(log(finalBoxWidth / initialBoxWidth) / (totalNumberOfImages - 1))
        maxTimeFactor = exp(log(finalMaxTimeToSimulate / initialMaxTimeToSimulate) / (totalNumberOfImages - 1))

        # This is used to avoid simulating pendulums that have already gone chaotic. It is the max amount
        # of time to simulate per kernel run before running the kernel to compute the chaos amount, at which
        # time pendulums that have gone chaotic are marked so they don't continue to be simulated.
        simulationTimeBetweenComputingChaosAmount = 2**3

        logger.info('Generating images for zoom sequence...')
        logger.info('fps = ' + str(fps))
        logger.info('total zoom time seconds = ' + str(totalZoomTimeSeconds))
        logger.info('total number of images = ' + str(totalNumberOfImages))
        logger.info('initial box width = ' + str(initialBoxWidth))
        logger.info('initial max time to simulate seconds = ' + str(initialMaxTimeToSimulate))
        logger.info('final box width = ' + str(finalBoxWidth))
        logger.info('box center = (' + str(boxCenterX) + ', ' + str(boxCenterY) + ')')
        logger.info('final max time to simulate seconds = ' + str(finalMaxTimeToSimulate))
        logger.info('zoom factor = ' + str(zoomFactor))
        logger.info('max time factor = ' + str(maxTimeFactor))
        logger.info('difference cutoff = ' + str(differenceCutoff))
        logger.info('simulationTimeBetweenComputingChaosAmount seconds = ' + str(simulationTimeBetweenComputingChaosAmount))

        # If the zoom factor is 1 then a zoom isn't being done, so to optimize the pendulums do not
        # have to be simulated from scratch for every image.
        currentStates = None
        chaosAmountData = None
        if zoomFactor == 1:
            currentStates = np.zeros((4, app.simulator.numberOfAnglesToTestY, app.simulator.numberOfAnglesToTestX), np.dtype(app.simulator.npFloatType))
            chaosAmountData = -1*np.ones((app.simulator.numberOfAnglesToTestY, app.simulator.numberOfAnglesToTestX), np.dtype(app.simulator.npFloatType))

        # Generate the images
        timeAlreadySimulated = 0
        for i in range(totalNumberOfImages):
            logger.info('Generating image ' + str(i + 1) + ' of ' + str(totalNumberOfImages))

            # Compute the zoomed-in bounding box
            newWidth = initialBoxWidth*zoomFactor**i
            self.simulator.set_angle1_min(boxCenterX - newWidth/2)
            self.simulator.set_angle1_max(boxCenterX + newWidth/2)
            self.simulator.set_angle2_min(boxCenterY - newWidth/2)
            self.simulator.set_angle2_max(boxCenterY + newWidth/2)

            # Simulate the pendulums
            curMaxTimeToSimulate = initialMaxTimeToSimulate*maxTimeFactor**i
            if zoomFactor != 1:
                image = self.generate_chaos_amount_image_from_scratch(curMaxTimeToSimulate, simulationTimeBetweenComputingChaosAmount, differenceCutoff)
            else:
                image = self.generate_chaos_amount_image(currentStates, chaosAmountData, curMaxTimeToSimulate, simulationTimeBetweenComputingChaosAmount, differenceCutoff, timeAlreadySimulated)
            imageFileName = f'{i:04}' + '.png'
            image.save(os.path.join(self.directoryToSaveData, imageFileName))
            timeAlreadySimulated = curMaxTimeToSimulate

            logger.info('')

        # Convert the images into an mp4 video
        videoName = 'chaos amount zoom video_' + str(boxCenterX) + '-' + str(boxCenterY)
        logger.info('Creating video from images...')
        create_video_from_images(self.directoryToSaveData, videoName, fps)
        logger.info('Finished creating video ' + videoName)


if __name__ == "__main__":

    # Configuration
    directoryToSaveData = create_directory()
    deviceNumberToUse = 0
    useDoublePrecision = True
    # algorithm = SimulationAlgorithm.RK_4
    # algorithm = SimulationAlgorithm.RKF_45
    # algorithm = SimulationAlgorithm.CASH_KARP_45
    # algorithm = SimulationAlgorithm.DORMAND_PRINCE_54
    algorithm = SimulationAlgorithm.FEHLBERG_87
    simulator = DoublePendulumCudaSimulator(deviceNumberToUse, directoryToSaveData, useDoublePrecision, algorithm, None)

    # The dimensions of the image in pixels.
    simulator.set_image_width_pixels(int(1000/2**0))
    simulator.set_image_height_pixels(simulator.imageResolutionWidthPixels)

    # The amount of super-sampling anti-aliasing to apply to the image. Can be fractional.
    # 1 means no anti-aliasing.
    # 2 means four total samples are used per pixel.
    # 3 means nine total samples are used per pixel, etc.
    simulator.set_anti_aliasing_amount(1)

    # Simulation parameters.
    simulator.set_time_step(.01/2**2)
    simulator.set_error_tolerance(1e-11)
    simulator.set_gravity(1)
    simulator.set_point1_mass(1)
    simulator.set_point2_mass(1)
    simulator.set_pendulum1_length(1)
    simulator.set_pendulum2_length(1)

    # Create the object to generate images.
    app = GenerateDoublePendulumFractalImages(simulator)

    # Generate a chaos amount zoom in video.
    app.generate_zoom_in_video_chaos_amount_method()



