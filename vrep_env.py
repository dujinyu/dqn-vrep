# -*- coding: utf-8 -*-
try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import sys
import numpy
import time
import math

from utils.fmu import FMU
from utils.geometry import rotate
from utils import taskspec

# rl-glue library
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal

class vrep_environment(Environment):
    # range of state space observations
    MAX_POS_XY = 0.5             # [m] maximum deviation in position in each dimension
    MAX_VEL = 1.0                 # [m/s] maximum velocity in each dimension
    MAX_LIN_RATE = 2.5           # [m/s] maximum velocity in each dimension            
    MAX_ANG_RATE = 4 * numpy.pi  # maximum angular velocity
    MAX_ANG = numpy.pi
    state_goal = numpy.array([0.0, 0.0, 0.5])
    state_ranges = numpy.array([[-MAX_POS_XY, MAX_POS_XY]] * 2
                             + [[-0.5, 0.5]] * 1
                             + [[-MAX_LIN_RATE, MAX_LIN_RATE]] * 3           
                             + [[-MAX_ANG_RATE, MAX_ANG_RATE]] * 3
                             + [[-MAX_ANG, MAX_ANG]] * 3)
    
    def env_init(self):
        print ('VREP Environmental Program Started')
        vrep.simxFinish(-1) # just in case, close all opened connections
        self.clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP
        if self.clientID!=-1:
            print ('Connected to remote API server')
            self.fmu = FMU()
            self.start_simulation() 
        else:
            print('Connection Failure')
            sys.exit('Abort Connection')
            
        return self.makeTaskSpec()
        
    def env_start(self):
        self.restart_simulation()
        returnObs=Observation()
        returnObs.doubleArray=self.getState()
        return returnObs
        
    def env_step(self,thisAction):
        # validate the action 
        assert len(thisAction.intArray)==1,"Expected 1 integer actions."
        
        self.takeAction(thisAction.intArray[0])
        
        theObs = Observation()
        theObs.doubleArray = self.getState()
        
        theReward,terminate = self.getReward()
        returnRO = Reward_observation_terminal()
        returnRO.r = theReward
        returnRO.o = theObs
        returnRO.terminal = int(terminate)

        return returnRO
        
    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        return "I got nothing to say about your message"
    
    def makeTaskSpec(self):
        ts = taskspec.TaskSpec(discount_factor=1,
                               reward_range=('UNSPEC','UNSPEC'))
        ts.setDiscountFactor(self.discount_factor)
        ts.addDiscreteAction((0, 7))
        for minValue, maxValue in self.state_ranges:
            ts.addContinuousObservation((minValue, maxValue))
        ts.setEpisodic()
        ts.setExtra("name: VREP quadcopter environment.")
        return ts.toTaskSpec()
        
    def start_simulation(self):
        mode = vrep.simx_opmode_oneshot_wait
        assert vrep.simxStartSimulation(self.clientID, mode) == 0, \
                                        "StartSim Error"
        assert vrep.simxSynchronous(self.clientID, True) == 0, \
                                        "Sync Error"
        self.config_handles()
        
    def restart_simulation(self):
        mode = vrep.simx_opmode_oneshot_wait
        assert vrep.simxStopSimulation(self.clientID, mode) == 0, \
                                       "StopSim Error"
        time.sleep(0.5)
        self.start_simulation()
        
    def proceed_simulation(self, sim_steps=1):
        for t in range(sim_steps):
            vrep.simxSynchronousTrigger(self.clientID)
        
    def config_handles(self):
        # Object-handle-configuration                                  
        errorFlag, self.Quadbase = vrep.simxGetObjectHandle(self.clientID,
                                                            'Quadricopter_base',
                                                            vrep.simx_opmode_oneshot_wait)
        errorFlag, self.Quadobj = vrep.simxGetObjectHandle(self.clientID,
                                                           'Quadricopter',
                                                           vrep.simx_opmode_oneshot_wait)
        self.getState(initial=True)
        time.sleep(1.5)
        
    def getState(self, initial=False):
        if initial:
            mode = vrep.simx_opmode_streaming
        else:
            mode = vrep.simx_opmode_buffer
            
        # Retrieve IMU data
        errorSignal, self.stepSeconds = vrep.simxGetFloatSignal(self.clientID,
                                                                'stepSeconds', mode)      
        errorOrien, baseEuler = vrep.simxGetObjectOrientation(self.clientID,
                                                              self.Quadbase, -1, mode) 
        errorPos, basePos = vrep.simxGetObjectPosition(self.clientID,
                                                       self.Quadbase,-1, mode)
        errorVel, linVel, angVel = vrep.simxGetObjectVelocity(self.clientID,
                                                              self.Quadbase, mode)         
                                                               
        if initial:
            if (errorSignal or errorOrien or errorPos or errorVel != vrep.simx_return_ok):
                time.sleep(.5)
            pass
        else:
            # Position error to desire
            basePos[0] = basePos[0]-self.state_goal[0]
            basePos[1] = basePos[1]-self.state_goal[1]
            basePos[2] = basePos[2]-self.state_goal[2]
            
            # Convert Euler angles to pitch, roll, yaw
            rollRad, pitchRad = rotate((baseEuler[0], baseEuler[1]), baseEuler[2])
            pitchRad = -pitchRad
            yawRad   = -baseEuler[2]
        
            baseRad = numpy.array([yawRad,rollRad,pitchRad])+0.0
                                                             
            self.state = numpy.concatenate((basePos, linVel, angVel, baseRad))
#            print("data_core: " + str(state[4]) + " " + str(state[5]) + " " + \
#                                  str(state[6]) + " " + str(state[7]) + " " + \
#                                  str(state[8]) + " " + str(state[9]))
            return self.state

    def takeAction(self,intAction):
        """
        state = (basePos[0], basePos[1], basePos[2],
                 linVel[0], linVel[1], linVel[2],
                 angVel[0], angVel[1], angVel[2],
                 baseYaw[0], baseRoll[1], basePitch[2])
        """
        action = numpy.array(numpy.zeros(4))
        if intAction == 0:
            action[0] = self.MAX_VEL
        if intAction == 1:
            action[0] = -self.MAX_VEL
        if intAction == 2:
            action[1] = self.MAX_VEL
        if intAction == 3:
            action[1] = -self.MAX_VEL
        if intAction == 4:
            action[2] = self.MAX_VEL
        if intAction == 5:
            action[2] = -self.MAX_VEL
        if intAction == 6:
            action[3] = self.MAX_VEL
        if intAction == 7:
            action[3] = -self.MAX_VEL
        
        # Get altitude directly from position Z
        altiMeters = self.state[2]+self.state_goal[2]
    
        # Get motor thrusts from FMU model
        thrusts = self.fmu.getMotors((self.state[11], self.state[10], self.state[9]), altiMeters,
                                     action, self.stepSeconds)
#        print("thrusts: " + str(thrusts[0]) + " " + str(thrusts[1]) + " " + \
#                str(thrusts[2]) + " " + str(thrusts[3]))
        #vrep.simxPauseCommunication(self.clientID,True)
        for t in range(4):
            errorFlag = vrep.simxSetFloatSignal(self.clientID,'thrusts'+str(t+1),
                                                thrusts[t], vrep.simx_opmode_oneshot)                                 
        #vrep.simxPauseCommunication(self.clientID,False)
        
        self.proceed_simulation()

    def getReward(self):
        r = math.exp(-numpy.sum(abs(self.state[:3])/(.1*(self.state_ranges[:3,1]-self.state_ranges[:3,0]))))
        terminate = False

        #if (x < -4.5 or x > 4.5  or theta < -twelve_degrees or theta > twelve_degrees):
        if numpy.any(self.state_ranges[:, 0] > self.state[:]) or \
           numpy.any(self.state_ranges[:, 1] < self.state[:]):
            r = -1
            terminate = True;

        return r,terminate
		

if __name__=="__main__":
	EnvironmentLoader.loadEnvironment(vrep_environment())