from state_agent import StateAgent
import sys,os
import numpy as np
import cv2

weights_path = "./agent_weights.h5"
data_file = "./data/rollout_Pong-v0/rollout-1.npz"
data = np.load(data_file)
next_states = data['next']
correct = data['correct']

print(next_states.shape)
print(correct.shape)

agent = StateAgent(6,'Pong-v0')
agent.set_weights(weights_path)

i = 5

current_next = next_states[i]
current_correct = correct[i]

cv2.imwrite("testimg.jpg",current_next[:,:,1]*255.)
print(current_correct)

current_correct = np.expand_dims(current_correct,axis=0)
current_next = np.expand_dims(current_next,axis=0)

prediction = agent.predict(current_next)
print(prediction.shape)
print(prediction)

