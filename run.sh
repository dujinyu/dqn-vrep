#!/bin/bash

# Find existed process ID to kill
PID=`ps -ef | grep rl_glue | grep -v grep | awk '{print $2}' | xargs kill`
if [[ "" != "$PID" ]]; then
    echo "kill $PID"
    kill -9 $PID
fi

workPath=~/py_ws/dqn/chainer/vrep

#Start RL-glue
rl_glue &

#Start Agent
python $workPath/agent_dqn_ddac.py &

#Start Environment
python $workPath/env_vrep.py &

#Start Experiment
python $workPath/exp_vrep.py &
