import pyglet
from pyglet.gl import *
from pyglet.window import key
from decimal import Decimal
import sys
import numpy as np
from win32api import GetSystemMetrics

import primitives
import pyglet_utils
from numerical_routines import *


class PrimaryWindow(pyglet.window.Window):

    FPS = 60
    timeStep = .01/2**2
    numStepsToComputePerFrame = int(ceil((1/FPS)/timeStep))
    smoothConfig = pyglet_utils.get_smooth_config()
    labelNumDecimalPlaces = Decimal(10) ** -12

    screenWidth = GetSystemMetrics(0)
    screenHeight = GetSystemMetrics(1)
    windowWidthPixels = int(screenWidth/2)
    windowHeightPixels = int(screenHeight/2)

    pixelsPerMeter = 100
    ballRadiusPixels = 20
    ballColor = (0, 128/255, 255/255, 1)
    gravity = 9.81
    point1Mass = 1
    point2Mass = 1
    pendulum1Length = 1
    pendulum2Length = 1
    point1AngularVelocity = 0
    point2AngularVelocity = 0

    # point1Angle = -pi+.1
    # point2Angle = 0
    point1Angle = (-3.371910665006095 - -3.396454357612266)/2 + -3.396454357612266
    point2Angle = (1.925992646191392 - 1.901448953585222)/2 + 1.901448953585222

    origin = [0, 0]
    originPixels = [origin[0]*pixelsPerMeter + windowWidthPixels/2, origin[1]*pixelsPerMeter + windowHeightPixels/2]

    simulationTime = 0

    minimumEnergyNeededForFlip = point1Mass*pendulum1Length*gravity + point2Mass*(pendulum1Length - pendulum2Length)*gravity;
    print('minimum energy needed to flip = ' + str(minimumEnergyNeededForFlip))

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

        # Compute the next steps of the simulation.
        for i in range(self.numStepsToComputePerFrame):
            self.point1Angle, \
            self.point2Angle, \
            self.point1AngularVelocity, \
            self.point2AngularVelocity = compute_double_pendulum_step_rk4(self.point1Mass, self.point2Mass,
                                                                          self.gravity,
                                                                          self.pendulum1Length, self.pendulum2Length,
                                                                          self.point1Angle, self.point2Angle,
                                                                          self.point1AngularVelocity, self.point2AngularVelocity,
                                                                          self.timeStep)


        self.simulationTime += self.timeStep*self.numStepsToComputePerFrame

        # Compute and display the energy of the system.
        totalEnergy = get_total_energy_of_pendulum(self.origin,
                                                   self.point1Angle, self.point2Angle,
                                                   self.point1AngularVelocity, self.point2AngularVelocity,
                                                   self.pendulum1Length, self.pendulum2Length,
                                                   self.point1Mass, self.point2Mass,
                                                   self.gravity)
        self.energyLabel.text = str(Decimal(totalEnergy).quantize(self.labelNumDecimalPlaces))

        
        # Update the position of the masses.
        point1Position = get_point_position(self.origin, self.point1Angle, self.pendulum1Length)
        point2Position = get_point_position(point1Position, self.point2Angle, self.pendulum2Length)

        point1PositionPixels = (point1Position[0]*self.pixelsPerMeter + self.windowWidthPixels/2, point1Position[1]*self.pixelsPerMeter + self.windowHeightPixels/2)
        point2PositionPixels = (point2Position[0]*self.pixelsPerMeter + self.windowWidthPixels/2, point2Position[1]*self.pixelsPerMeter + self.windowHeightPixels/2)

        self.point1.x = point1PositionPixels[0]
        self.point1.y = point1PositionPixels[1]
        self.point2.x = point2PositionPixels[0]
        self.point2.y = point2PositionPixels[1]

        self.line1 = primitives.Line(self.originPixels, point1PositionPixels, stroke=5, color=(255,255,255,1))
        self.line2 = primitives.Line(point1PositionPixels, point2PositionPixels, stroke=5, color=(255,255,255,1))






def get_point_position(origin, angle, pendulumLength):
    x = sin(angle)*pendulumLength + origin[0]
    y = -cos(angle)*pendulumLength + origin[1]

    return [x, y]


def get_point_position_np(origin, angle, pendulumLength):
    x = np.sin(angle)*pendulumLength + origin[0]
    y = -np.cos(angle)*pendulumLength + origin[1]

    return [x, y]


def get_total_energy_of_pendulum(origin, 
                                 point1Angle, point2Angle,
                                 point1AngularVelocity, point2AngularVelocity,
                                 pendulum1Length, pendulum2Length,
                                 point1Mass, point2Mass,
                                 gravity):
    
    point1Position = get_point_position(origin, point1Angle, pendulum1Length)
    point2Position = get_point_position(point1Position, point2Angle, pendulum2Length)

    # Compute the potential energy of the masses.
    potentialEnergy1 = point1Position[1]*point1Mass*gravity
    potentialEnergy2 = point2Position[1]*point2Mass*gravity
    
    # Compute the kinetic energy of the first mass.
    point1Velocity = pendulum1Length*point1AngularVelocity
    kineticEnergyPoint1 = .5*point1Mass*point1Velocity**2

    # Compute the kinetic energy of the second mass.
    point1VelocityX = cos(point1Angle)*point1Velocity
    point1VelocityY = sin(point1Angle)*point1Velocity
    point2LocalVelocity = pendulum2Length*point2AngularVelocity
    point2VelocityX = cos(point2Angle)*point2LocalVelocity + point1VelocityX
    point2VelocityY = sin(point2Angle)*point2LocalVelocity + point1VelocityY
    point2Velocity = sqrt(point2VelocityX**2 + point2VelocityY**2)
    kineticEnergyPoint2 = .5*point2Mass*point2Velocity**2

    totalEnergy = potentialEnergy1 + potentialEnergy2 + kineticEnergyPoint1 + kineticEnergyPoint2

    return totalEnergy


def get_total_energy_of_pendulum_np(origin,
                                    point1Angle, point2Angle,
                                    point1AngularVelocity, point2AngularVelocity,
                                    pendulum1Length, pendulum2Length,
                                    point1Mass, point2Mass,
                                    gravity):

    point1Position = get_point_position_np(origin, point1Angle, pendulum1Length)
    point2Position = get_point_position_np(point1Position, point2Angle, pendulum2Length)

    # Compute the potential energy of the masses.
    potentialEnergy1 = point1Position[1]*point1Mass*gravity
    potentialEnergy2 = point2Position[1]*point2Mass*gravity

    # Compute the kinetic energy of the first mass.
    point1Velocity = pendulum1Length*point1AngularVelocity
    kineticEnergyPoint1 = .5*point1Mass*point1Velocity**2

    # Compute the kinetic energy of the second mass.
    point1VelocityX = np.cos(point1Angle)*point1Velocity
    point1VelocityY = np.sin(point1Angle)*point1Velocity
    point2LocalVelocity = pendulum2Length*point2AngularVelocity
    point2VelocityX = np.cos(point2Angle)*point2LocalVelocity + point1VelocityX
    point2VelocityY = np.sin(point2Angle)*point2LocalVelocity + point1VelocityY
    point2Velocity = sqrt(point2VelocityX**2 + point2VelocityY**2)
    kineticEnergyPoint2 = .5*point2Mass*point2Velocity**2

    totalEnergy = np.float32(potentialEnergy1 + potentialEnergy2 + kineticEnergyPoint1 + kineticEnergyPoint2)

    return totalEnergy






if __name__ == "__main__":
    PrimaryWindow()