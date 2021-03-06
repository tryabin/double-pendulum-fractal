import enum
import logging
import os
import sys
import time

import numpy as np
import pycuda.driver as cuda
from PIL import Image
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt

from utils import read_file

logger = logging.getLogger(__name__)

class SimulationAlgorithm(enum.Enum):
   RK_4 = 1
   RKF_45 = 2
   CASH_KARP_45 = 3
   DORMAND_PRINCE_54 = 4
   FEHLBERG_87 = 5

ADAPTIVE_STEP_SIZE_METHODS = [SimulationAlgorithm.RKF_45, SimulationAlgorithm.CASH_KARP_45, SimulationAlgorithm.DORMAND_PRINCE_54, SimulationAlgorithm.FEHLBERG_87]

class DoublePendulumCudaSimulator:

    # The amount of super-sampling anti-aliasing to apply to the image. Can be fractional.
    # 1 means no anti-aliasing.
    # 2 means four total samples are used per pixel.
    # 3 means nine total samples are used per pixel, etc.
    antiAliasingAmount = 1

    # Default optimal max register counts for the kernels
    optimalMaxRegisterCountTimeTillFlipKernel = 80
    optimalMexRegisterCountAmountOfChaosKernel = 76

    def __init__(self, deviceNumberToUse, directoryToSaveData, useDoublePrecision, algorithm, maxRegistersToUse):

        # The Fehlberg 8(7) doesn't work well with 32-bit floating point precision.
        if not useDoublePrecision and algorithm is SimulationAlgorithm.FEHLBERG_87:
            sys.exit('Can\'t use Fehlberg 8(7) algorithm with 32-bit precision, not enough precision for 8th order method, exiting...')

        # Initialize the CUDA driver.
        cuda.init()
        self.device = cuda.Device(deviceNumberToUse)
        self.device.make_context()

        # Initialize the logger.
        self.directoryToSaveData = directoryToSaveData
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(self.directoryToSaveData + '/log.log'))
        logger.info('GPU being used: ' + self.device.name())

        # Configure the options to send to the nvcc compiler.
        options = ['-DFLOAT_64'] if useDoublePrecision else ['-DFLOAT_32']
        # The Runge-Kutta algorithm to use in the kernel.
        if algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            options.append('-D' + str(algorithm.name))

        # The max register count, used for performance optimization.
        timeTillFlipKernelOptions = options.copy()
        amountOfChaosKernelOptions = options.copy()
        if useDoublePrecision:
            if maxRegistersToUse is not None:
                timeTillFlipKernelOptions.append('-maxrregcount=' + str(maxRegistersToUse))
                amountOfChaosKernelOptions.append('-maxrregcount=' + str(maxRegistersToUse))
            else:
                timeTillFlipKernelOptions.append('-maxrregcount=' + str(self.optimalMaxRegisterCountTimeTillFlipKernel))
                amountOfChaosKernelOptions.append('-maxrregcount=' + str(self.optimalMexRegisterCountAmountOfChaosKernel))

        logger.info('options = ' + str(options))
        logger.info('timeTillFlipKernelOptions = ' + str(timeTillFlipKernelOptions))
        logger.info('amountOfChaosKernelOptions = ' + str(amountOfChaosKernelOptions))

        # Initialize the kernels.
        self.npFloatType = np.float64 if useDoublePrecision else np.float32
        includeDir = os.getcwd() + '/src/cuda/include'
        self.algorithm = algorithm
        if algorithm is SimulationAlgorithm.RK_4:
            timeTillFlipMethodKernelFile = 'src/cuda/rk4.cu'
            self.computeDoublePendulumFractalFromInitialStatesRK4Function = SourceModule(read_file(timeTillFlipMethodKernelFile), include_dirs=[includeDir], options=timeTillFlipKernelOptions).get_function('compute_double_pendulum_fractal_steps_till_flip_from_initial_states')
            self.computeColorsFromStepsTillFlip = SourceModule(read_file(timeTillFlipMethodKernelFile), include_dirs=[includeDir], options=options).get_function('compute_colors_from_steps_till_flip')
        elif algorithm in ADAPTIVE_STEP_SIZE_METHODS:
            timeTillFlipMethodKernelFile = 'src/cuda/compute_double_pendulum_fractal_time_till_flip_method.cu'
            self.computeDoublePendulumFractalWithTimeTillFlipMethodAndAdaptiveStepSize = SourceModule(read_file(timeTillFlipMethodKernelFile), include_dirs=[includeDir], options=timeTillFlipKernelOptions).get_function('compute_double_pendulum_fractal_time_till_flip_from_initial_states')
            self.computeColorsFromTimeTillFlip = SourceModule(read_file(timeTillFlipMethodKernelFile), include_dirs=[includeDir], options=options).get_function('compute_colors_from_time_till_flip')

            amountOfChaosMethodKernelFile = 'src/cuda/compute_double_pendulum_fractal_amount_of_chaos_method.cu'
            self.computeDoublePendulumFractalWithAmountOfChaosMethod = SourceModule(read_file(amountOfChaosMethodKernelFile), include_dirs=[includeDir], options=amountOfChaosKernelOptions).get_function('compute_double_pendulum_fractal_amount_of_chaos_method')
            self.computeAmountOfChaos = SourceModule(read_file(amountOfChaosMethodKernelFile), include_dirs=[includeDir], options=options).get_function('compute_amount_of_chaos')


    def get_directory_to_save_data(self):
        return self.directoryToSaveData

    def set_directory_to_save_data(self, directoryToSaveData):
        self.directoryToSaveData = directoryToSaveData

    def set_angle1_min(self, value):
        self.angle1Min = value

    def set_angle1_max(self, value):
        self.angle1Max = value

    def set_angle2_min(self, value):
        self.angle2Min = value

    def set_angle2_max(self, value):
        self.angle2Max = value

    def set_image_width_pixels(self, width):
        self.imageResolutionWidthPixels = width
        self.numberOfAnglesToTestX = int(round(self.imageResolutionWidthPixels*self.antiAliasingAmount))

    def set_image_height_pixels(self, height):
        self.imageResolutionHeightPixels = height
        self.numberOfAnglesToTestY = int(round(self.imageResolutionHeightPixels*self.antiAliasingAmount))

    def set_image_dimensions_based_on_width(self, width):
        self.imageResolutionWidthPixels = int(width)
        self.imageResolutionHeightPixels = int(round(self.imageResolutionWidthPixels*(self.angle2Max - self.angle2Min)/(self.angle1Max - self.angle1Min)))
        self.set_number_of_angles_to_test()

    def set_number_of_angles_to_test(self):
        if self.imageResolutionWidthPixels is not None:
            self.numberOfAnglesToTestX = int(round(self.imageResolutionWidthPixels*self.antiAliasingAmount))
        if self.imageResolutionHeightPixels is not None:
            self.numberOfAnglesToTestY = int(round(self.imageResolutionHeightPixels*self.antiAliasingAmount))

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
        image = imageWithoutAntiAliasing.resize((self.imageResolutionWidthPixels, self.imageResolutionHeightPixels), Image.LANCZOS)
        logger.info('Finished creating image')

        return image


    def compute_new_pendulum_states_time_till_flip_adaptive_step_size_method(self, currentStates, timeTillFlipData, timeAlreadyExecuted, maxTimeToExecute, startFromDefaultState):
        logger.info('Computing new pendulum states with ' + str(self.algorithm.name) + ' method')
        logger.info('Using the "time till flip" kernel')
        logger.info('time step: ' + str(self.timeStep) + ' seconds')
        logger.info('error tolerance: ' + str(self.errorTolerance))
        logger.info('amount of time already computed: ' + str(timeAlreadyExecuted) + ' seconds')
        logger.info('max time to see if pendulum flips: ' + str(maxTimeToExecute) + ' seconds')
        logger.info('amount of time to simulate: ' + str(maxTimeToExecute - timeAlreadyExecuted) + ' seconds')

        # Compute the double pendulum fractal image.
        logger.info('Running pendulum simulation kernel...')
        kernelStart = time.time()

        self.computeDoublePendulumFractalWithTimeTillFlipMethodAndAdaptiveStepSize(self.npFloatType(self.point1Mass), self.npFloatType(self.point2Mass),
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
        image = imageWithoutAntiAliasing.resize((self.imageResolutionWidthPixels, self.imageResolutionHeightPixels), Image.LANCZOS)
        logger.info('Finished creating image')

        return image


    def compute_new_pendulum_states_amount_of_chaos_adaptive_step_size_method(self, currentStates, amountOfChaos, timeAlreadyExecuted, maxTimeToExecute, startFromDefaultState):
        logger.info('Computing new pendulum states with ' + str(self.algorithm.name) + ' method')
        logger.info('Using the "amount of chaos" kernel')
        logger.info('time step: ' + str(self.timeStep) + ' seconds')
        logger.info('error tolerance: ' + str(self.errorTolerance))
        logger.info('amount of time already computed: ' + str(timeAlreadyExecuted) + ' seconds')
        logger.info('max time to simulate: ' + str(maxTimeToExecute) + ' seconds')
        logger.info('amount of time to simulate: ' + str(maxTimeToExecute - timeAlreadyExecuted) + ' seconds')

        # Compute the double pendulum fractal image.
        logger.info('Running pendulum simulation kernel...')
        kernelStart = time.time()

        self.computeDoublePendulumFractalWithAmountOfChaosMethod(self.npFloatType(self.point1Mass), self.npFloatType(self.point2Mass),
                                                                 self.npFloatType(self.pendulum1Length), self.npFloatType(self.pendulum2Length),
                                                                 self.npFloatType(self.gravity),
                                                                 self.npFloatType(self.angle1Min), self.npFloatType(self.angle1Max),
                                                                 self.npFloatType(self.angle2Min), self.npFloatType(self.angle2Max),
                                                                 cuda.InOut(currentStates),
                                                                 cuda.In(amountOfChaos),
                                                                 np.int32(startFromDefaultState),
                                                                 self.npFloatType(timeAlreadyExecuted),
                                                                 np.int32(self.numberOfAnglesToTestX), np.int32(self.numberOfAnglesToTestY),
                                                                 self.npFloatType(self.timeStep),
                                                                 self.npFloatType(self.errorTolerance),
                                                                 self.npFloatType(maxTimeToExecute),
                                                                 # block=(1, 1, 1), grid=(1, 1))
                                                                 # block=(2, 2, 1), grid=(1, 1))
                                                                 # block=(4, 4, 1), grid=(4, 4))
                                                                 # block=(8, 8, 1), grid=(8, 8))
                                                                 block=(16, 16, 1), grid=(16, 16))
                                                                 # block=(32, 32, 1), grid=(32, 32))

        # Print the time it took to run the kernel.
        timeToExecuteLastKernel = time.time() - kernelStart
        logger.info('Completed pendulum simulation kernel in ' + str(timeToExecuteLastKernel) + ' seconds')


    def compute_chaos_amount_from_pendulum_states(self, currentStates, amountOfChaos, differenceCutoff):
        logger.info('Computing chaos amount data from pendulum states...')
        logger.info('differenceCutoff = ' + str(differenceCutoff))

        # Run a kernel to compute the chaos amount for each pendulum.
        start = time.time()
        self.computeAmountOfChaos(cuda.In(currentStates),
                                  cuda.InOut(amountOfChaos),
                                  np.int32(self.numberOfAnglesToTestX),
                                  np.int32(self.numberOfAnglesToTestY),
                                  self.npFloatType(differenceCutoff),
                                  block=(16, 16, 1), grid=(16, 16))
        logger.info('Computed chaos amount kernel in ' + str(time.time() - start) + ' seconds')


    def create_image_from_amount_of_chaos(self, currentStates, amountOfChaos, differenceCutoff):
        logger.info('Creating image from time till flip...')
        logger.info('differenceCutoff = ' + str(differenceCutoff))

        # Run a kernel to compute the colors from the time step counts.
        self.computeAmountOfChaos(cuda.In(currentStates),
                                  cuda.InOut(amountOfChaos),
                                  np.int32(self.numberOfAnglesToTestX),
                                  np.int32(self.numberOfAnglesToTestY),
                                  self.npFloatType(differenceCutoff),
                                  block=(16, 16, 1), grid=(16, 16))

        # Remove the border rows and columns from the amount of chaos array.
        amountOfChaosWithBordersRemoved = np.delete(amountOfChaos, (0, self.numberOfAnglesToTestX-1), 0)
        amountOfChaosWithBordersRemoved = np.delete(amountOfChaosWithBordersRemoved, (0, self.numberOfAnglesToTestY-1), 1)

        logger.info('max stability value = ' + str(np.amax(amountOfChaosWithBordersRemoved)))

        # Create an image from the amount of chaos.
        fig, ax = plt.subplots(figsize=((self.numberOfAnglesToTestX - 2)/100, (self.numberOfAnglesToTestY - 2)/100), frameon=False)

        # Use matplotlib to color the image like a heatmap. Add vmin and vmax to keep the coloring consistent between zoom images.
        plt.imshow(amountOfChaosWithBordersRemoved, cmap='hot', interpolation='nearest')
        # plt.imshow(amountOfChaosWithBordersRemoved, cmap='hot', interpolation='nearest', vmin=0, vmax=100)

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        w, h, d = buf.shape
        imageWithoutAntiAliasing = Image.frombytes("RGBA", (w, h), buf.tobytes())
        image = imageWithoutAntiAliasing.resize((self.imageResolutionWidthPixels - 2, self.imageResolutionHeightPixels - 2), Image.LANCZOS)
        plt.close()

        logger.info('Finished creating image')

        return image