from calicoEnv import *
from policies import ActionDecoder, TransformerBuffer
import tqdm
import torch.nn as nn
import torch.optim as optim
from numpy import random
import matplotlib.pyplot as plt
# if self.training:
#     distribution = torch.distributions.Categorical(logits=action['a1'])
#     action1 = distribution.sample()
#     distribution = torch.distributions.Categorical(logits=action['a2'])
#     action2 = distribution.sample()
#     distribution = torch.distributions.Categorical(logits=action['a3'])
#     action3 = distribution.sample()
# else:
#     action1 = torch.argmax(action['a1'])
#     action2 = torch.argmax(action['a2'])
#     action3 = torch.argmax(action['a3'])

def collect_trajectories(env, model, desired_performance, num_episodes=100):
    model.eval()
    env.eval()
    trajectories = []
    cycle_score = []
    for _ in range(num_episodes):
        goal_performance = random.randint(desired_performance[0],desired_performance[1])
        obs, info = env.reset(goal_performance)
        episode = {'observations': [], 'actions': []}#, 'returns': [], 'action_mask':[]}
        total_reward = 0
        done = False
        while not done:
            # obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action_logits = model(obs, info['action_mask'])  # You may need to adapt input formatting
                action1 = torch.argmax(action_logits['a1'])
                action2 = torch.argmax(action_logits['a2'])
                action3 = torch.argmax(action_logits['a3'])
                # print(action_logits['a1'])
                # print(action1,action2,action3)

            next_obs, reward, done, _, info = env.step([action1,action2,action3])

            episode['observations'] = obs
            # print(obs['return_to_go'])
            episode['actions'].append([action1,action2,action3])
            # episode['returns'].append(reward)
            # episode['action_mask'].append(info['action_mask'])
            total_reward = reward
            episode['observations']['return_to_go'][-1] = goal_performance-total_reward
            obs = next_obs
        # print(episode['observations']['return_to_go'], total_reward)
        # episode['returns'] = compute_discounted_returns(episode['returns'])
        # print('what the fuck', episode['returns'])
        # if episode['observations']['return_to_go'][-1] >0:
        episode['observations']['return_to_go'] = episode['observations']['return_to_go']-episode['observations']['return_to_go'][-1]
        # print(episode['observations']['return_to_go'])
        trajectories.append((episode, total_reward))
        cycle_score.append(total_reward)
    cycle_score = np.average(cycle_score)

    return trajectories, cycle_score

def compute_discounted_returns(rewards, gamma=1.0):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_model(env, model, buffer, num_epochs=10, batch_size=64, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    average_scores =[]
    average_losses = []
    # prime the replay buffer
    trajectories, start_score = collect_trajectories(env, model, [5,15], num_episodes=100)
    for t in trajectories:
        buffer.add(t[0],t[1])
    average_scores.append(start_score)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        trajectories, cycle_score = collect_trajectories(env, model, [buffer.getAverage()[0],buffer.getMax()], num_episodes=100)
        for t in trajectories:
            buffer.add(t[0],t[1])
        # obs_tensor, action_tensor, return_tensor, action_mask_tensor = prepare_dataset(trajectories)
        print(f"Average Score {cycle_score:.3f}")
        model.train()
        env.train()

        total_loss = 0
        average_scores.append(cycle_score)
        total_1_loss = 0
        total_2_loss = 0
        total_3_loss = 0
        # print('about to start the learning')
        for i in range(10):
            batch = buffer.sample_batch(batch_size)
            # print(batch)
            # print(type(batch))
            obs_batch = batch['observations'] #.to(device)
            action_batch = batch['actions'] #.to(device)
            # action_mask_batch = batch['action_mask'] #.to(device)
            # print()
            # board_thing, neighbor_thing, store_thing, hand_thing = obs_batch['state'].getTensor()
            # return_batch = batch['returns'] #.to(device)
            # Model forward
            logits = model(obs_batch)#,action_mask_batch)  # Ensure this works with your transformer structure

            a1loss = criterion(logits['a1'], action_batch[:,0])
            a2loss = criterion(logits['a2'], action_batch[:,1])
            a3loss = criterion(logits['a3'], action_batch[:,2])
            net_loss = a1loss+a2loss+a3loss
            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()
            total_loss += net_loss.item()
            total_1_loss += a1loss.item()
            total_2_loss += a2loss.item()
            total_3_loss += a3loss.item()
        m, st = buffer.getAverage()
        print(f"Avg Loss:    {total_loss / 10:.3f}")
        print(f"Avg A1 Loss: {total_1_loss / 10:.3f}")
        print(f"Avg A2 Loss: {total_2_loss / 10:.3f}")
        print(f"Avg A3 Loss: {total_3_loss / 10:.3f}")
        print('best performing run: ', buffer.getMax())
        print(f'buffer average score: {m:0.3f} +/- {st:0.3f}')
        print(f'buffer size: {len(buffer)}')
        average_losses.append(total_loss)
    return model, {'score':average_scores, 'loss':average_losses}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CalicoEnv()
model = ActionDecoder(128,4)

replay_buffer = TransformerBuffer(8000)

trained_model, info = train_model(env,model,replay_buffer,1000)

plt.plot(range(len(info['score'])),info['score'])
# plt.plot(range(len(info['score'])),info['score'])
plt.show()
torch.save(trained_model.state_dict(), 'trained_calico.pt')