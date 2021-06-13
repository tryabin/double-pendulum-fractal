import logging
import sys
import time
from math import pi
from pathlib import Path

import winsound

import numpy as np

from double_pendulum_kernel_methods import DoublePendulumCudaSimulator, SimulationAlgorithm, ADAPTIVE_STEP_SIZE_METHODS

logger = logging.getLogger(__name__)

class DoublePendulumFractalPerformanceTest:

    # Configuration
    deviceNumberToUse = 0
    useDoublePrecision = True
    # algorithm = SimulationAlgorithm.RK_4
    # algorithm = SimulationAlgorithm.RKF_45
    # algorithm = SimulationAlgorithm.CASH_KARP_45
    # algorithm = SimulationAlgorithm.DORMAND_PRINCE_54
    algorithm = SimulationAlgorithm.FEHLBERG_87

    def __init__(self, directoryToSaveData):
        # The directory used to store the image and pendulum data files.
        self.directoryToSaveData = directoryToSaveData
        Path(directoryToSaveData).mkdir(parents=True, exist_ok=True)

        # Initialize the logger.
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.FileHandler(self.directoryToSaveData + '/log.log'))


    def initialize_simulator(self, maxRegisterCount=80):
        self.simulator = DoublePendulumCudaSimulator(self.deviceNumberToUse, self.directoryToSaveData, self.useDoublePrecision, self.algorithm, maxRegisterCount)

        # The range of pendulum angles.

        # Enough energy to flip first mass area
        # self.simulator.set_angle1_min(-3/2*pi)
        # self.simulator.set_angle1_max(-1/2*pi)
        # self.simulator.set_angle2_min(0*pi)
        # self.simulator.set_angle2_max(2*pi)

        # 7 o'clock island of stability
        # self.simulator.set_angle1_min(-3.396454357612266)
        # self.simulator.set_angle1_max(-3.371910665006095)
        # self.simulator.set_angle2_min(1.901448953585222)
        # self.simulator.set_angle2_max(1.925992646191392)

        # Square enough energy to flip first mass area
        self.simulator.set_angle1_min(-2*pi)
        self.simulator.set_angle1_max(0)
        self.simulator.set_angle2_min(0)
        self.simulator.set_angle2_max(2*pi)

        # The width of the image in pixels.
        self.simulator.set_image_dimensions_based_on_width(int(1000))

        # The amount of super-sampling anti-aliasing to apply to the image. Can be fractional.
        # 1 means no anti-aliasing.
        # 2 means four total samples are used per pixel.
        # 3 means nine total samples are used per pixel, etc.
        self.simulator.set_anti_aliasing_amount(1)

        # Simulation parameters.
        self.simulator.set_time_step(.01/2**2)
        self.simulator.set_error_tolerance(1e-7)
        self.simulator.set_gravity(1)
        self.simulator.set_point1_mass(1)
        self.simulator.set_point2_mass(1)
        self.simulator.set_pendulum1_length(1)
        self.simulator.set_pendulum2_length(1)


    def run_kernel_performance_test(self, numTimesToRunKernel, maxTimeToExecute):
        initialStates = np.zeros((4, self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))
        numTimeStepsTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(np.int32))
        timeTillFlip = np.zeros((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), self.simulator.npFloatType)
        amountOfChaos = -1 * np.ones((self.simulator.numberOfAnglesToTestY, self.simulator.numberOfAnglesToTestX), np.dtype(self.simulator.npFloatType))
        maxTimeStepsToExecute = maxTimeToExecute/self.simulator.timeStep

        # Run the kernel the given number of times and compute the average kernel run time.
        totalTime = 0
        for i in range(numTimesToRunKernel):
            start = time.time()

            # Run the kernel corresponding to the simulation algorithm to use.
            if self.algorithm is SimulationAlgorithm.RK_4:
                self.simulator.compute_new_pendulum_states_rk4(initialStates, numTimeStepsTillFlip, 0, maxTimeStepsToExecute, True)
            elif self.algorithm in ADAPTIVE_STEP_SIZE_METHODS:
                self.simulator.compute_new_pendulum_states_amount_of_chaos_adaptive_step_size_method(initialStates, amountOfChaos, 0, maxTimeToExecute, True)
                # self.simulator.compute_new_pendulum_states_time_till_flip_adaptive_step_size_method(initialStates, timeTillFlip, 0, maxTimeToExecute, True)

            totalTime += time.time() - start

        return totalTime / numTimesToRunKernel


if __name__ == "__main__":
    startingMaxRegisterCount = 76
    endingMaxRegisterCount = 76
    numTimesToRunKernel = 1
    maxTimeSteps = 2**6
    app = DoublePendulumFractalPerformanceTest('./tmp')

    logger.info('maxRegisterCount, average kernel time (seconds)')
    for i in range(startingMaxRegisterCount, endingMaxRegisterCount + 1):
        app.initialize_simulator(i)
        averageKernelTime = app.run_kernel_performance_test(numTimesToRunKernel, maxTimeSteps)
        logger.info(str(i) + ',' + str(averageKernelTime))
        # logger.info('average kernel time = ' + str(averageKernelTime))

    winsound.Beep(1500, 1000)