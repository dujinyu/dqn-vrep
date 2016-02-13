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
import time
import math
import numpy as np

from vrepUtils.fmu import FMU
from vrepUtils.geometry import rotate
from vrepUtils import taskspec

# rl-glue library
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal

class vrep_environment(Environment):
    # range of state space observations
    MAX_POS_XY = 0.5             # [m] maximum deviation in position in each dimension
    MAX_LIN_RATE = 1.0           # [m/s] maximum velocity in each dimension            
    MAX_ANG_RATE = 4 * np.pi     # [rad/s] maximum angular velocity
    MAX_ANG = np.pi * 30./180.   # [rad] maximum angular
    state_goal = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    state_ranges = np.array([[-MAX_POS_XY, MAX_POS_XY]] * 2
                            + [[0.0, 1.0]] * 1
                            + [[-MAX_LIN_RATE, MAX_LIN_RATE]] * 3           
                            + [[-MAX_ANG_RATE, MAX_ANG_RATE]] * 3
                            + [[-MAX_ANG, MAX_ANG]] * 3)
    action_ranges = np.array([[-1., 1.]] * 2)
    discount_factor=.99
    
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
        returnObs.doubleArray=self.getState().tolist()
        return returnObs
        
    def env_step(self,thisAction):
        # validate the action 
        assert len(thisAction.doubleArray)==2,"Expected 4 double actions."
        
        self.takeAction(thisAction.doubleArray)
        
        theObs = Observation()
        theObs.doubleArray = self.getState().tolist()
        
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
        ts = taskspec.TaskSpec(discount_factor=self.discount_factor,
                               reward_range=('UNSPEC','UNSPEC'))
        ts.setDiscountFactor(self.discount_factor)
        for minValue, maxValue in self.action_ranges:        
            ts.addContinuousAction((minValue, maxValue))
        for minValue, maxValue in self.state_ranges:
            ts.addContinuousObservation((minValue, maxValue))
        ts.setEpisodic()
        ts.setExtra("name: VREP quadcopter environment.")
        return ts.toTaskSpec()
        
    def start_simulation(self):
        mode = vrep.simx_opmode_oneshot_wait
        assert vrep.simxStartSimulation(self.clientID, mode) == 0,"StartSim Error"
        assert vrep.simxSynchronous(self.clientID, True) == 0,"Sync Error"
        self.config_handles()
        
    def restart_simulation(self):
        mode = vrep.simx_opmode_oneshot_wait
        assert vrep.simxStopSimulation(self.clientID, mode) == 0, \
                                       "StopSim Error"
        time.sleep(0.1)
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
        time.sleep(0.05)
        
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
                time.sleep(0.05)
            pass
        else:       
            # Convert Euler angles to pitch, roll, yaw
            rollRad, pitchRad = rotate((baseEuler[0], baseEuler[1]), baseEuler[2])
            pitchRad = -pitchRad
            yawRad   = -baseEuler[2]
        
            baseRad = np.array([yawRad,rollRad,pitchRad])+0.0   
            self.state = np.asarray(np.concatenate((basePos,linVel,angVel,baseRad)),dtype=np.float32)
            #print("data_core: " + str(self.state))
            return self.state

    def takeAction(self,doubleAction):
        """
        state = (basePos[0], basePos[1], basePos[2],
                 linVel[3], linVel[4], linVel[5],
                 angVel[6], angVel[7], angVel[8],
                 baseYaw[9], baseRoll[10], basePitch[11])
        """
#        self.prevState = self.state
        #print("oldState: "+str(self.state))
        action = np.asarray(doubleAction,dtype=np.float32)
        #print("takeAction: "+str(doubleAction))
        
        # Get altitude directly from position Z
        altiMeters = self.state[2]
        
        # Get motor thrusts from FMU model
        thrusts = self.fmu.getMotors((self.state[11],self.state[10],self.state[9]),altiMeters,
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
        #print("newState: "+str(self.state))
        r = 0.0
        if np.any(self.state_ranges[:,0] > self.state[:]) or \
           np.any(self.state_ranges[:,1] < self.state[:]):
#            r = -1
            r = -np.sum(3.0 * self.state_ranges[:,1]**2)
            terminate = True
        else:
#            perr = np.linalg.norm(self.prevState[:2] - self.state_goal[:2])
#            nerr = np.linalg.norm(self.state[:2] - self.state_goal[:2])
#            r = math.exp(-np.sum(abs(self.state[:2]-self.state_goal[:2])/(.1*(self.state_ranges[:2,1]-self.state_ranges[:2,0]))))* \
#                math.exp(-np.sum(abs(self.state[3:5]-self.state_goal[3:5])/(.1*(self.state_ranges[3:5,1]-self.state_ranges[3:5,0]))))* \
#                math.exp(-np.sum(abs(self.state[6:8]-self.state_goal[6:8])/(.1*(self.state_ranges[6:8,1]-self.state_ranges[6:8,0]))))
#            r = math.exp(-np.sum(abs(self.state[:2]-self.state_goal[:2])/(.1*(self.state_ranges[:2,1]-self.state_ranges[:2,0]))))* \
#                math.exp(-np.sum(abs(self.state[3:5]-self.state_goal[3:5])/(.1*(self.state_ranges[3:5,1]-self.state_ranges[3:5,0]))))
#            r -= (np.sum(((self.state[:2]-self.state_goal[:2])/(self.state_ranges[:2,1]-self.state_ranges[:2,0]))**2)+ \
#                  np.sum(((self.state[3:5]-self.state_goal[3:5])/(self.state_ranges[3:5,1]-self.state_ranges[3:5,0]))**2))
            r -= (self.state[0]-self.state_goal[0])**2
            r -= (self.state[1]-self.state_goal[1])**2
            r -= self.state[3]**2
            r -= self.state[4]**2
            
            terminate = False

        print("reward "+str(r))
        return r,terminate
		
if __name__=="__main__":
	EnvironmentLoader.loadEnvironment(vrep_environment())
