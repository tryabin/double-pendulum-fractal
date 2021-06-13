import sys

import pyglet
from pyglet.gl import *
from pyglet.window import key
from win32api import GetSystemMetrics

import primitives
import pyglet_utils
from numerical_routines import *
from numerical_routines import get_point_position, get_total_energy_of_pendulum


class PrimaryWindow(pyglet.window.Window):

    # Basic config
    FPS = 60
    timeStep = .01/2**2
    errorTolerance = 1e-7
    mp.prec = 53
    numStepsToComputePerFrame = int(ceil((1/FPS)/timeStep))
    simulationAlgorithm = SimulationAlgorithm.RK_4
    # simulationAlgorithm = SimulationAlgorithm.RKF_45
    # simulationAlgorithm = SimulationAlgorithm.DORMAND_PRINCE_54
    # simulationAlgorithm = SimulationAlgorithm.VERNER_65
    # simulationAlgorithm = SimulationAlgorithm.FEHLBERG_87
    print('algorithm = ' + str(simulationAlgorithm.name))

    # UI config
    screenWidth = GetSystemMetrics(0)
    screenHeight = GetSystemMetrics(1)
    windowWidthPixels = int(screenWidth/2)
    windowHeightPixels = int(screenHeight/2)
    smoothConfig = pyglet_utils.get_smooth_config()
    labelNumDecimalPlaces = 20
    pixelsPerMeter = 100
    ballRadiusPixels = 20
    ballColor = (0, 128/255, 255/255, 1)
    origin = [0, 0]
    originPixels = [origin[0]*pixelsPerMeter + windowWidthPixels/2, origin[1]*pixelsPerMeter + windowHeightPixels/2]

    # Simulation config
    gravity = 9.81
    point1Mass = 1
    point2Mass = 1
    pendulum1Length = 1
    pendulum2Length = 1
    point1AngularVelocity = 0
    point2AngularVelocity = 0

    # 7 o'clock "Big Three" stable area
    # minAngle1 = -3.396454357612266
    # maxAngle1 = -3.371910665006095
    # minAngle2 = 1.901448953585222
    # maxAngle2 = 1.925992646191392

    # 9 o'clock "Big Three" stable area
    # minAngle1 = -4.0033707883776435
    # maxAngle1 = -3.9051960179529623
    # minAngle2 = 3.2218996157971826
    # maxAngle2 = 3.3200743862218642

    # 11 o'clock "Big Three" stable area
    # minAngle1 = -3.6061556672393835
    # maxAngle1 = -3.507980896814703
    # minAngle2 = 4.390964782014284
    # maxAngle2 = 4.489139552438965

    # 10 o'clock little brother to "Big Three" stable area
    # minAngle1 = -4.106454297323556
    # maxAngle1 = -4.008279526898876
    # minAngle2 = 4.120198765183015
    # maxAngle2 = 4.218373535607696

    # 2nd mass spins in one direction
    minAngle1 = 2.850507319679923
    maxAngle1 = 2.8535752812556945
    minAngle2 = 2.10120065625985
    maxAngle2 = 2.1042686178356216

    # 7 o'clock edge stable area
    # minAngle1 = -4.169727936862266
    # maxAngle1 = -4.157456090559181
    # minAngle2 = 1.6173557116687927
    # maxAngle2 = 1.6296275579718777

    # 8 o'clock just beyond edge stable area
    # minAngle1 = -4.535968917931539
    # maxAngle1 = -4.486881532719199
    # minAngle2 = 1.9843084598236533
    # maxAngle2 = 2.0333958450359937

    # Just outside first mass flip energy near 45 degree big branch
    # minAngle1 = -8.145757051676634
    # maxAngle1 = -7.949407510827273
    # minAngle2 = -2.1080086705587506
    # maxAngle2 = -1.9116591297093883

    # 5 o'clock high energy stable area
    # minAngle1 = -2.7491880962023423
    # maxAngle1 = -2.7369162498992567
    # minAngle2 = 2.5978026002074843
    # maxAngle2 = 2.61007444651057

    # Edge of flip area 6 o'clock
    # minAngle1 = -3.176346522320131
    # maxAngle1 = -3.1272591371077896
    # minAngle2 = 0.17318029502913748
    # maxAngle2 = 0.222267680241478

    # Deep high energy 2
    # minAngle1 = -2.9295305475301197
    # maxAngle1 = -2.929434673730877
    # minAngle2 = 3.253602396740415
    # maxAngle2 = 3.2536982705396578

    point1Angle = (maxAngle1 - minAngle1)/2 + minAngle1
    point2Angle = (maxAngle2 - minAngle2)/2 + minAngle2

    print('point1Angle = ' + str(point1Angle))
    print('point2Angle = ' + str(point2Angle))
    minimumEnergyNeededForFlip = point1Mass*pendulum1Length*gravity + point2Mass*(pendulum1Length - pendulum2Length)*gravity
    simulationTime = 0


    def __init__(self):
        super(PrimaryWindow, self).__init__(config=self.smoothConfig)
        self.set_caption('Double Pendulum')
        self.set_size(self.windowWidthPixels, self.windowHeightPixels)
        self.set_location(int(self.screenWidth/2 - self.windowWidthPixels/2), int(self.screenHeight/2 - self.windowHeightPixels/2))

        self.point1 = primitives.Circle(width=self.ballRadiusPixels*2, color=self.ballColor)
        self.point2 = primitives.Circle(width=self.ballRadiusPixels*2, color=self.ballColor)
        self.line1 = primitives.Line()
        self.line2 = primitives.Line()

        self.energyLabel = pyglet.text.Label(font_name='Times New Roman',
                                             font_size=36,
                                             x=self.windowWidthPixels/2, y=self.windowHeightPixels/2 + 200,
                                             anchor_x='center', anchor_y='center')

        pyglet.clock.schedule_interval(self.update, 1.0/self.FPS)
        pyglet.app.run()


    def on_draw(self):
        self.clear()
        self.line1.render()
        self.line2.render()
        self.point1.render()
        self.point2.render()
        self.energyLabel.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            sys.exit(0)

    def update(self, dt):

        # Simulate the simulation forward one frame.
        amountOfTimeSimulatedForFrame = 0
        timeStepToUseInCalculation = self.timeStep
        while amountOfTimeSimulatedForFrame < 1/self.FPS:
            # Lower the time step to hit the frame rate on the last step to simulate for the frame.
            timeStepToUseInCalculation = min(1/self.FPS - amountOfTimeSimulatedForFrame, timeStepToUseInCalculation)

            # RK4
            if self.simulationAlgorithm is SimulationAlgorithm.RK_4:
                self.point1Angle, \
                self.point2Angle, \
                self.point1AngularVelocity, \
                self.point2AngularVelocity = compute_double_pendulum_step_rk4(self.point1Mass, self.point2Mass,
                                                                              self.gravity,
                                                                              self.pendulum1Length, self.pendulum2Length,
                                                                              self.point1Angle, self.point2Angle,
                                                                              self.point1AngularVelocity, self.point2AngularVelocity,
                                                                              timeStepToUseInCalculation)
                self.simulationTime += timeStepToUseInCalculation
                amountOfTimeSimulatedForFrame += timeStepToUseInCalculation

            # Adaptive time step methods
            elif self.simulationAlgorithm in ADAPTIVE_TIME_STEP_METHODS:
                self.point1Angle, \
                self.point2Angle, \
                self.point1AngularVelocity, \
                self.point2AngularVelocity, \
                timeStepUsedInCalculation, newTimeStep = compute_double_pendulum_step_with_adaptive_step_size_method(self.point1Mass, self.point2Mass,
                                                                                                                     self.gravity,
                                                                                                                     self.pendulum1Length, self.pendulum2Length,
                                                                                                                     self.point1Angle, self.point2Angle,
                                                                                                                     self.point1AngularVelocity, self.point2AngularVelocity,
                                                                                                                     timeStepToUseInCalculation,
                                                                                                                     self.errorTolerance,
                                                                                                                     self.simulationAlgorithm)
                timeStepToUseInCalculation = newTimeStep
                self.simulationTime += timeStepUsedInCalculation
                amountOfTimeSimulatedForFrame += timeStepUsedInCalculation
                self.timeStep = timeStepToUseInCalculation

        # Compute and display the energy of the system.
        totalEnergy = get_total_energy_of_pendulum(self.origin,
                                                   self.point1Angle, self.point2Angle,
                                                   self.point1AngularVelocity, self.point2AngularVelocity,
                                                   self.pendulum1Length, self.pendulum2Length,
                                                   self.point1Mass, self.point2Mass,
                                                   self.gravity)
        self.energyLabel.text = str(mp.nstr(totalEnergy, self.labelNumDecimalPlaces, strip_zeros=False))
        
        # Update the position of the masses.
        point1Position = get_point_position(self.origin, self.point1Angle, self.pendulum1Length)
        point2Position = get_point_position(point1Position, self.point2Angle, self.pendulum2Length)

        point1PositionPixels = (point1Position[0]*self.pixelsPerMeter + self.windowWidthPixels/2, point1Position[1]*self.pixelsPerMeter + self.windowHeightPixels/2)
        point2PositionPixels = (point2Position[0]*self.pixelsPerMeter + self.windowWidthPixels/2, point2Position[1]*self.pixelsPerMeter + self.windowHeightPixels/2)

        self.point1.x = point1PositionPixels[0]
        self.point1.y = point1PositionPixels[1]
        self.point2.x = point2PositionPixels[0]
        self.point2.y = point2PositionPixels[1]

        self.line1 = primitives.Line(self.originPixels, point1PositionPixels, stroke=5, color=(255, 255, 255, 1))
        self.line2 = primitives.Line(point1PositionPixels, point2PositionPixels, stroke=5, color=(255, 255, 255, 1))


if __name__ == "__main__":
    PrimaryWindow()