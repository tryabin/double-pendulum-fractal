import pyglet
from decimal import Decimal
from pyglet.gl import *
from math import *

import primitives
import pyglet_utils
from numerical_routines import *


class PrimaryWindow(pyglet.window.Window):

    FPS = 60
    timestep = .01
    numStepsToComputePerFrame = int(ceil((1/FPS)/timestep))
    smoothConfig = pyglet_utils.get_smooth_config()

    screenWidth = pyglet.window.get_platform().get_default_display().get_default_screen().width
    screenHeight = pyglet.window.get_platform().get_default_display().get_default_screen().height

    pixelsPerMeter = 100
    ballRadiusPixels = 20
    ballColor = (0, 128/255, 255/255, 1)
    gravity = 9.81
    point1Mass = 10
    point2Mass = 10
    pendulum1Length = 1
    pendulum2Length = 1
    point1AngularVelocity = 0
    point2AngularVelocity = 0

    point1Angle = pi
    point2Angle = 0

    origin = [0, 2]
    originPixels = [origin[0]*pixelsPerMeter + screenWidth/2, origin[1]*pixelsPerMeter + screenHeight/2]

    simulationTime = 0

    def __init__(self):
        super(PrimaryWindow, self).__init__(config=self.smoothConfig, style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS)
        self.set_caption('Double Pendulum')
        self.set_size(self.screenWidth, self.screenHeight)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.point1 = primitives.Circle(width=self.ballRadiusPixels*2, color=self.ballColor)
        self.point2 = primitives.Circle(width=self.ballRadiusPixels*2, color=self.ballColor)
        self.line1 = primitives.Line()
        self.line2 = primitives.Line()

        self.energyLabel = pyglet.text.Label(font_name='Times New Roman',
                                             font_size=36,
                                             x=self.screenWidth/2, y=self.screenHeight/2 + 400,
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


    def update(self, dt):

        point1Position = get_point_position(self.origin, self.point1Angle, self.pendulum1Length)
        point2Position = get_point_position(point1Position, self.point2Angle, self.pendulum2Length)

        # Compute the potential energy of the masses.
        potentialEnergy1 = point1Position[1]*self.point1Mass*self.gravity
        potentialEnergy2 = point2Position[1]*self.point2Mass*self.gravity
        # potentialEnergy1 = (point1Position[1] - (self.origin[1] - self.pendulum1Length))*self.point1Mass*self.gravity
        # potentialEnergy2 = (point2Position[1] - (self.origin[1] - self.pendulum1Length - self.pendulum2Length))*self.point2Mass*self.gravity

        # Compute the kinetic energy of the first mass.
        point1Velocity = self.pendulum1Length*self.point1AngularVelocity
        kineticEnergyPoint1 = .5*self.point1Mass*point1Velocity**2

        # Compute the kinetic energy of the second mass.
        point1VelocityX = cos(self.point1Angle)*point1Velocity
        point1VelocityY = sin(self.point1Angle)*point1Velocity
        point2LocalVelocity = self.pendulum2Length*self.point2AngularVelocity
        point2VelocityX = cos(self.point2Angle)*point2LocalVelocity + point1VelocityX
        point2VelocityY = sin(self.point2Angle)*point2LocalVelocity + point1VelocityY
        kineticEnergyPoint2 = .5*self.point2Mass*(point2VelocityX**2 + point2VelocityY**2)
        totalEnergy = potentialEnergy1 + potentialEnergy2 + kineticEnergyPoint1 + kineticEnergyPoint2


        # Compute the next steps of the simulation.
        for i in range(self.numStepsToComputePerFrame):
            self.point1AngularVelocity, self.point2AngularVelocity, self.point1Angle, self.point2Angle = compute_double_pendulum_step_rk4(self.point1Mass, self.point2Mass,
                                                                                                                                          self.gravity,
                                                                                                                                          self.pendulum1Length, self.pendulum2Length,
                                                                                                                                          self.point1AngularVelocity, self.point2AngularVelocity,
                                                                                                                                          self.point1Angle, self.point2Angle,
                                                                                                                                          self.timestep)

            # self.point1AngularVelocity, self.point2AngularVelocity, self.point1Angle, self.point2Angle = compute_double_pendulum_step_euler(self.point1Mass, self.point2Mass,
            #                                                                                                                             self.gravity,
            #                                                                                                                             self.pendulum1Length, self.pendulum2Length,
            #                                                                                                                             self.point1AngularVelocity, self.point2AngularVelocity,
            #                                                                                                                             self.point1Angle, self.point2Angle,
            #                                                                                                                             self.timestep)

        self.simulationTime += self.timestep*self.numStepsToComputePerFrame
        TWOPLACES = Decimal(10) ** -2
        self.energyLabel.text = str(Decimal(totalEnergy).quantize(TWOPLACES))
        # self.energyLabel.text = str(Decimal(self.simulationTime).quantize(TWOPLACES))


        # Update the position of the masses.
        point1PositionPixels = (point1Position[0]*self.pixelsPerMeter + self.screenWidth/2, point1Position[1]*self.pixelsPerMeter + self.screenHeight/2)
        point2PositionPixels = (point2Position[0]*self.pixelsPerMeter + self.screenWidth/2, point2Position[1]*self.pixelsPerMeter + self.screenHeight/2)

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





if __name__ == "__main__":
    PrimaryWindow()