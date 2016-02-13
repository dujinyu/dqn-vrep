# -*- coding: utf-8 -*-
"""
Simple RL glue experiment setup
"""
import json
import numpy as np
import rlglue.RLGlue as RLGlue

max_Episode = 20000
max_validEpisode = 7
pass_validation = False
whichEpisode = 0
learningEpisode = 0

def runEpisode(is_learning_episode):
    global whichEpisode, learningEpisode

    RLGlue.RL_episode(10000)
    totalSteps = RLGlue.RL_num_steps()
    totalReward = RLGlue.RL_return()

    whichEpisode += 1   

    if is_learning_episode:
        learningEpisode += 1
        #print "Episode " + str(learningEpisode) + "/" + str(whichEpisode) + "\t " + str(totalSteps) + " steps \t" + str(totalReward) + " total reward\t "
        print "Episode %d/%d\t %d steps \t %.1f total reward\t" % (learningEpisode,whichEpisode,totalSteps,totalReward)
    else:
        #print "Evaluation ::\t " + str(totalSteps) + " steps \t" + str(totalReward) + " total reward\t "
        print "Evaluation ::\t %d steps \t %.1f total reward" % (totalSteps,totalReward)
        with open('eval_dump.json', 'a') as f:
            json.dump({"Steps": totalSteps, "Episode": whichEpisode, "Reward": totalReward}, f)
            f.write('\n')
        return totalSteps
        
# Main Program starts here
print "\n\nDQN-ALE Experiment starting up!"
RLGlue.RL_init()

while whichEpisode < max_Episode:
       
    # Evaluate model every 10 episodes
    if (whichEpisode >= 250 and np.mod(whichEpisode, 10) == 0) or pass_validation:
        print "Freeze learning for Evaluation"
        RLGlue.RL_agent_message("freeze learning")
        if runEpisode(is_learning_episode=False) >= 6000:
            print "Freeze learning for Validation"
            validEpisode = 0
            # Validate model for early stopping
            for i in range(max_validEpisode):
                RLGlue.RL_agent_message("freeze learning")
                if runEpisode(is_learning_episode=False) >= 6000:
                    validEpisode += 1
            if validEpisode >= max_validEpisode:
                pass_validation = True
            else:
                pass_validation = False
    else:
        print "DQN is Learning"
        RLGlue.RL_agent_message("unfreeze learning")
        runEpisode(is_learning_episode=True)

    # Save model every 100 learning episodes
    if np.mod(learningEpisode, 100) == 0 and learningEpisode != 0:
        print "SAVE CURRENT MODEL"
        RLGlue.RL_agent_message("save model")
            
RLGlue.RL_cleanup()

print "Experiment COMPLETED @ Episode ", whichEpisode
