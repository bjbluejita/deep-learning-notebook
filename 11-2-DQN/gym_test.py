'''
@Project: deep-learning-with-keras-notebooks
@Package 
@author: ly
@date Date: 2019年11月14日 16:09
@Description: 
@URL: 
@version: V1.0
'''
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()