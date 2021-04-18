import enum
import logging
import os
import sys
import time

import numpy as np
import pycuda.driver as cuda
from PIL import Image
from pycuda.compiler import SourceModule

from utils import read_file

logger = logging.getLogger('root')


class SimulationAlgorithm(enum.Enum):
   RK4 = 1
   RKF45 = 2
   CASH_KARP = 3

ADAPTIVE_STEP_SIZE_METHODS = [SimulationAlgorithm.RKF45, SimulationAlgorithm.CASH_KARP]

class DoublePendulumCudaSimulator:

    # The amount of super-sampling anti-aliasing to apply to the image. Can be fractional.
    # 1 means no anti-aliasing.
    # 2 means four total samples are used per pixel.
    # 3 means nine total samples are used per pixel, etc.
    antiAliasingAmount = 1

    def __init__(self, deviceNumberToUse, directoryToSaveData, useDoublePrecision, algorithm, maxRegistersToUse=80):
        # Initialize the CUDA driver.
        cuda.init()
        self.device = cuda.Device(deviceNumberToUse)
        self.device.make_context()

        # Initialize the logger.
        self.directoryToSaveData = directoryToSaveData
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(self.directoryToSaveData + '/log.log'))
        logger.info('GPU being used: ' + self.device.name())

        # Configure the options to send to the nvcc compiler.
        options = ['-DFLOAT_64'] if useDoublePrecision else ['-DFLOAT_32']
        if useDoublePrecision:
            options.append('-maxrregcount=' + str(maxRegistersToUse))
        if algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            options.append('-D' + str(algorithm.name))
        logger.info('options = ' + str(options))

        # Initialize the kernels.
        self.npFloatType = np.float64 if useDoublePrecision else np.float32
        includeDir = os.getcwd() + '/src/cuda/include'
        self.algorithm = algorithm
        if algorithm is SimulationAlgorithm.RK4:
            kernelFile = 'src/cuda/rk4.cu'
            self.computeDoublePendulumFractalFromInitialStatesRK4Function = SourceModule(read_file(kernelFile), include_dirs=[includeDir], options=options).get_function('compute_double_pendulum_fractal_steps_till_flip_from_initial_states')
            self.computeColorsFromStepsTillFlip = SourceModule(read_file(kernelFile), include_dirs=[includeDir], options=options).get_function('compute_colors_from_steps_till_flip')
        elif algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            kernelFile = 'src/cuda/adaptive_step_size_methods.cu'
            self.computeDoublePendulumFractalFromInitialStatesWithAdaptiveStepSizeFunction = SourceModule(read_file(kernelFile), include_dirs=[includeDir], options=options).get_function('compute_double_pendulum_fractal_time_till_flip_from_initial_states')
            self.computeColorsFromTimeTillFlip = SourceModule(read_file(kernelFile), include_dirs=[includeDir], options=options).get_function('compute_colors_from_time_till_flip')


    def set_angle1_min(self, value):
        self.angle1Min = value

    def set_angle1_max(self, value):
        self.angle1Max = value

    def set_angle2_min(self, value):
        self.angle2Min = value

    def set_angle2_max(self, value):
        self.angle2Max = value

    def set_image_width_pixels(self, width):
        self.imageResolutionPixelsWidth = width
        self.imageResolutionPixelsHeight = round(self.imageResolutionPixelsWidth*(self.angle2Max - self.angle2Min)/(self.angle1Max - self.angle1Min))
        self.set_number_of_angles_to_test()

    def set_number_of_angles_to_test(self):
        self.numberOfAnglesToTestX = int(round(self.imageResolutionPixelsWidth * self.antiAliasingAmount))
        self.numberOfAnglesToTestY = int(round(self.imageResolutionPixelsHeight * self.antiAliasingAmount))

    def set_time_step(self, value):
        self.timeStep = value

    def set_error_tolerance(self, value):
        self.errorTolerance = value

    def set_gravity(self, value):
        self.gravity = value

    def set_point1_mass(self, value):
        self.point1Mass = value

    def set_point2_mass(self, value):
        self.point2Mass = value

    def set_pendulum1_length(self, value):
        self.pendulum1Length = value

    def set_pendulum2_length(self, value):
        self.pendulum2Length = value

    def set_anti_aliasing_amount(self, value):
        self.antiAliasingAmount = value
        self.set_number_of_angles_to_test()


    def compute_new_pendulum_states_rk4(self, currentStates, numTimeStepsTillFlipData, numTimeStepsAlreadyExecuted, maxTimeStepsToExecute, startFromDefaultState):
        logger.info('Computing new pendulum states with Runge-Kutta 4th order method')
        logger.info('time step: ' + str(self.timeStep) + ' seconds')
        logger.info('amount of time already computed: ' + str(numTimeStepsAlreadyExecuted * self.timeStep) + ' seconds')
        logger.info('max time to see if pendulum flips: ' + str(maxTimeStepsToExecute * self.timeStep) + ' seconds')
        logger.info('amount of time to simulate: ' + str((maxTimeStepsToExecute - numTimeStepsAlreadyExecuted) * self.timeStep) + ' seconds')

        # Compute the double pendulum fractal image.
        logger.info('Running pendulum simulation kernel...')
        kernelStart = time.time()

        self.computeDoublePendulumFractalFromInitialStatesRK4Function(self.npFloatType(self.point1Mass), self.npFloatType(self.point2Mass),
                                                                      self.npFloatType(self.pendulum1Length), self.npFloatType(self.pendulum2Length),
                                                                      self.npFloatType(self.gravity),
                                                                      self.npFloatType(self.angle1Min), self.npFloatType(self.angle1Max),
                                                                      self.npFloatType(self.angle2Min), self.npFloatType(self.angle2Max),
                                                                      cuda.InOut(currentStates),
                                                                      np.int32(startFromDefaultState),
                                                                      np.int32(numTimeStepsAlreadyExecuted),
                                                                      np.int32(self.numberOfAnglesToTestX), np.int32(self.numberOfAnglesToTestY),
                                                                      self.npFloatType(self.timeStep),
                                                                      np.int32(maxTimeStepsToExecute),
                                                                      cuda.InOut(numTimeStepsTillFlipData),
                                                                      # block=(1, 1, 1), grid=(1, 1))
                                                                      # block=(2, 2, 1), grid=(1, 1))
                                                                      # block=(4, 4, 1), grid=(4, 4))
                                                                      # block=(8, 8, 1), grid=(8, 8))
                                                                      block=(16, 16, 1), grid=(16, 16))
                                                                      # block=(32, 32, 1), grid=(32, 32))

        # Print the time it took to run the kernel.
        timeToExecuteLastKernel = time.time() - kernelStart
        logger.info('Completed pendulum simulation kernel in ' + str(timeToExecuteLastKernel) + ' seconds')


    def create_image_from_number_of_time_steps_till_flip(self, numTimeStepsTillFlipData, redScale, greenScale, blueScale, shift):
        logger.info('Creating image from number of time steps till flip...')
        logger.info('redScale = ' + str(redScale))
        logger.info('greenScale = ' + str(greenScale))
        logger.info('blueScale = ' + str(blueScale))
        logger.info('shift = ' + str(shift))

        # Run a kernel to compute the colors from the time step counts.
        colors = np.zeros((3, self.numberOfAnglesToTestY, self.numberOfAnglesToTestX), np.dtype(np.uint8))
        self.computeColorsFromStepsTillFlip(cuda.In(numTimeStepsTillFlipData),
                                            cuda.Out(colors),
                                            np.int32(self.numberOfAnglesToTestX),
                                            np.int32(self.numberOfAnglesToTestY),
                                            self.npFloatType(self.timeStep),
                                            self.npFloatType(redScale),
                                            self.npFloatType(greenScale),
                                            self.npFloatType(blueScale),
                                            self.npFloatType(shift),
                                            block=(16, 16, 1), grid=(16, 16))

        # Create an image from the colors.
        redArray = Image.fromarray(colors[0])
        greenArray = Image.fromarray(colors[1])
        blueArray = Image.fromarray(colors[2])
        imageWithoutAntiAliasing = Image.merge('RGB', (redArray, greenArray, blueArray))
        image = imageWithoutAntiAliasing.resize((self.imageResolutionPixelsWidth, self.imageResolutionPixelsHeight), Image.LANCZOS)
        logger.info('Finished creating image')

        return image


    def compute_new_pendulum_states_runge_kutta_adaptive_step_size(self, currentStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, startFromDefaultState):
        logger.info('Computing new pendulum states with ' + str(self.algorithm.name) + ' method')
        logger.info('time step: ' + str(self.timeStep) + ' seconds')
        logger.info('error tolerance: ' + str(self.errorTolerance))
        logger.info('amount of time already computed: ' + str(timeAlreadyExecuted) + ' seconds')
        logger.info('max time to see if pendulum flips: ' + str(maxTimeToExecute) + ' seconds')
        logger.info('amount of time to simulate: ' + str(maxTimeToExecute - timeAlreadyExecuted) + ' seconds')

        # Compute the double pendulum fractal image.
        logger.info('Running pendulum simulation kernel...')
        kernelStart = time.time()

        self.computeDoublePendulumFractalFromInitialStatesWithAdaptiveStepSizeFunction(self.npFloatType(self.point1Mass), self.npFloatType(self.point2Mass),
                                                                                       self.npFloatType(self.pendulum1Length), self.npFloatType(self.pendulum2Length),
                                                                                       self.npFloatType(self.gravity),
                                                                                       self.npFloatType(self.angle1Min), self.npFloatType(self.angle1Max),
                                                                                       self.npFloatType(self.angle2Min), self.npFloatType(self.angle2Max),
                                                                                       cuda.InOut(currentStates),
                                                                                       np.int32(startFromDefaultState),
                                                                                       self.npFloatType(timeAlreadyExecuted),
                                                                                       np.int32(self.numberOfAnglesToTestX), np.int32(self.numberOfAnglesToTestY),
                                                                                       self.npFloatType(self.timeStep),
                                                                                       self.npFloatType(self.errorTolerance),
                                                                                       self.npFloatType(maxTimeToExecute),
                                                                                       cuda.InOut(timeTillFlipData),
                                                                                       # block=(1, 1, 1), grid=(1, 1))
                                                                                       # block=(2, 2, 1), grid=(1, 1))
                                                                                       # block=(4, 4, 1), grid=(4, 4))
                                                                                       # block=(8, 8, 1), grid=(8, 8))
                                                                                       block=(16, 16, 1), grid=(16, 16))
                                                                                       # block=(32, 32, 1), grid=(32, 32))

        # Print the time it took to run the kernel.
        timeToExecuteLastKernel = time.time() - kernelStart
        logger.info('Completed pendulum simulation kernel in ' + str(timeToExecuteLastKernel) + ' seconds')


    def create_image_from_time_till_flip(self, timeTillFlipData, redScale, greenScale, blueScale, shift):
        logger.info('Creating image from time till flip...')
        logger.info('redScale = ' + str(redScale))
        logger.info('greenScale = ' + str(greenScale))
        logger.info('blueScale = ' + str(blueScale))
        logger.info('shift = ' + str(shift))

        # Run a kernel to compute the colors from the time step counts.
        colors = np.zeros((3, self.numberOfAnglesToTestY, self.numberOfAnglesToTestX), np.dtype(np.uint8))
        self.computeColorsFromTimeTillFlip(cuda.In(timeTillFlipData),
                                            cuda.Out(colors),
                                            np.int32(self.numberOfAnglesToTestX),
                                            np.int32(self.numberOfAnglesToTestY),
                                            self.npFloatType(redScale),
                                            self.npFloatType(greenScale),
                                            self.npFloatType(blueScale),
                                            self.npFloatType(shift),
                                            block=(16, 16, 1), grid=(16, 16))

        # Create an image from the colors.
        redArray = Image.fromarray(colors[0])
        greenArray = Image.fromarray(colors[1])
        blueArray = Image.fromarray(colors[2])
        imageWithoutAntiAliasing = Image.merge('RGB', (redArray, greenArray, blueArray))
        image = imageWithoutAntiAliasing.resize((self.imageResolutionPixelsWidth, self.imageResolutionPixelsHeight), Image.LANCZOS)
        logger.info('Finished creating image')

        return image