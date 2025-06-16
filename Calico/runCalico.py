from calicoEnv import *
from policies import ActionDecoder



env = CalicoEnv()
model = ActionDecoder(128,4)

for i in range(100):
    obs, info = env.reset()
    playing = True
    while playing:
        action = model.forward(obs, info['action_mask'])
        obs, reward, done, _ , info = env.step(action)
        # if reward != 0:
        #     print(reward)
        playing = not done