'''
fmu.py - Flight Management Unit class

    Copyright (C) 2014 Bipeen Acharya, Fred Gisa, and Simon D. Levy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as 
    published by the Free Software Foundation, either version 3 of the 
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

'''

# PID parameters (I is currently unused) ==========================================

IMU_PITCH_ROLL_Kp       = .25   # original: .25
IMU_PITCH_ROLL_Kd       = 0.1

IMU_YAW_Kp 	            = 1.0
IMU_YAW_Kd 	            = 0.4

# We don't need K_d because we use first derivative
ALTITUDE_Kp             = 11

# Altitude hold z-pisition
Hold_Pos                = 0.51

# Empirical constants  ============================================================

THRUST_BASELINE 	       = 5.335
ROLL_DEMAND_FACTOR      = 0.1
PITCH_DEMAND_FACTOR     = 0.1
YAW_DEMAND_FACTOR       = 0.5
CLIMB_DEMAND_FACTOR     = 0.5

# Imports =========================================================================

from pidcontrol import Stability_PID_Controller, Yaw_PID_Controller, Hover_PID_Controller

# FMU class ==================================================================

class FMU(object):

    def __init__(self):
        '''
        Creates a new Quadrotor object.
        '''  
        # Create PD controllers for pitch, roll based on angles from Inertial Measurement Unit (IMU)
        self.pitch_Stability_PID = Stability_PID_Controller(IMU_PITCH_ROLL_Kp, IMU_PITCH_ROLL_Kd)      
        self.roll_Stability_PID  = Stability_PID_Controller(IMU_PITCH_ROLL_Kp, IMU_PITCH_ROLL_Kd)

        # Special handling for yaw from IMU
        self.yaw_IMU_PID   = Yaw_PID_Controller(IMU_YAW_Kp, IMU_YAW_Kd)

        # Create PD controller for altitude-hold
        self.altitude_PID = Hover_PID_Controller(ALTITUDE_Kp)

    def getMotors(self, imuAngles, altitude, controllerInput, timestep):
        '''
        Gets motor thrusts based on current telemetry:

            imuAngles      IMU pitch, roll, yaw angles in radians
            altitude       altitude in meters
            gpsCoords      GPS coordinates (latitude, longitude) in degrees
            controllInput  (pitchDemand, rollDemand, yawDemand, climbDemand, switch) 
            timestep       timestep in seconds
        '''
        # Convert flight-stick controllerInput
        pitchDemand = controllerInput[0] * PITCH_DEMAND_FACTOR
        rollDemand  = controllerInput[1] * ROLL_DEMAND_FACTOR
        yawDemand   = controllerInput[2] * YAW_DEMAND_FACTOR
        climbDemand = controllerInput[3] * CLIMB_DEMAND_FACTOR

        # Compute altitude hold if we want it
        altitudeHold = self.altitude_PID.getCorrection(altitude,timestep=timestep)

        # PID control for pitch, roll based on angles from Inertial Measurement Unit (IMU)
        imuPitchCorrection = self.pitch_Stability_PID.getCorrection(imuAngles[0], timestep)      
        imuRollCorrection  = self.roll_Stability_PID.getCorrection(-imuAngles[1], timestep)

        # Special PID for yaw
        yawCorrection   = self.yaw_IMU_PID.getCorrection(imuAngles[2], yawDemand, timestep)
              
        # Overall pitch, roll correction is sum of stability and position-hold 
        pitchCorrection = imuPitchCorrection
        rollCorrection  = imuRollCorrection
        
        # Overall thrust is baseline plus climb demand plus correction from PD controller
        thrust = THRUST_BASELINE + climbDemand + altitudeHold
        #print('thrust: ' + str(climbDemand) + ' ' + str(altitudeHold) + ' ' + str(altitude))
        
        # Change the thrust values depending upon the pitch, roll, yaw and climb values 
        # received from the joystick and the # quadrotor model corrections. A positive 
        # pitch value means, thrust increases for two back propellers and a negative 
        # is opposite; similarly for roll and yaw.  A positive climb value means thrust 
        # increases for all 4 propellers.

        psign = [+1, -1, -1, +1]    
        rsign = [-1, -1, +1, +1]
        ysign = [+1, -1, +1, -1]

        thrusts = [0]*4
 
        for i in range(4):
            thrusts[i] = (thrust + rsign[i]*rollDemand + psign[i]*pitchDemand + ysign[i]*yawDemand)*(1 + rsign[i]*rollCorrection + psign[i]*pitchCorrection + ysign[i]*yawCorrection) 

        return thrusts
