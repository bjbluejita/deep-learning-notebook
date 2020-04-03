'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年10月08日 12:16
@Description: 
@URL:https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_18/q_learning.py
@version: V1.0
'''
from __future__ import print_function
import numpy as np
import time
from env import Env

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_STEP = 30

np.random.seed( 43 )

def epsilon_greedy( Q, state ):
    if ( np.random.uniform() > 1 - EPSILON ) or ( ( Q[ state, :] == 0 ).all() ): #Q表当前状态还没记录采取4种action对应的回报（reward),从0-3中随机选一个
        action = np.random.randint( 0, 4 )  #0-3
    else:  #Q表当前状态已经记录采用action对应的回报（rewar),返回那个回报最大的action
        action = Q[ state, : ].argmax()
    return action

e = Env()
Q = np.zeros( ( e.state_num, 4 ) )

for i in range( 200 ):
    e = Env()
    e.print_map()
    while ( e.is_end is False ) and ( e.step < MAX_STEP ):
        action = epsilon_greedy( Q, e.present_state )
        state = e.present_state
        reward = e.interact( action )  #计算采用这个动作产生的回报值
        new_state = e.present_state
        Q[ state, action ] = ( 1 - ALPHA ) * Q[ state, action ] + \
            ALPHA * ( reward + GAMMA * Q[ new_state, :].max() )
        e.print_map()
        time.sleep( 0.1 )

    print( 'Episode:', i, 'Total Step:', e.step, 'Total Reward:', e.total_reward )
    time.sleep( 2 )

print( 'Q: ', Q )