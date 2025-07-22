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
def categorical_entropy(actions, num_classes):
    counts = torch.bincount(actions, minlength=num_classes).float()
    probs = counts / counts.sum()
    entropy = -torch.sum(probs * torch.log(probs + 1e-9))/np.log(num_classes)  # avoid log(0)
    return entropy.item()

def collect_trajectories(env, model, desired_performance, num_episodes=100, epsilon=0.1):
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
                if np.random.rand() > epsilon:
                    action_logits = model(obs, info['action_mask'])  # You may need to adapt input formatting
                    action1 = torch.argmax(action_logits['a1'])
                    action2 = torch.argmax(action_logits['a2'])
                    action3 = torch.argmax(action_logits['a3'])
                else:
                    action1 = torch.randint(0,3,())
                    action2 = torch.randint(0,3,())
                    valid_indices = np.nonzero(info['action_mask'])[0]
                    action3 = valid_indices[torch.randint(len(valid_indices), (1,))].item()
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


def train_model(env, model, buffer, num_epochs=10, batch_size=64, lr=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    average_scores =[]
    average_losses = []
    epsilon = 0.5
    estep = 0.5/400
    # prime the replay buffer
    trajectories, start_score = collect_trajectories(env, model, [5,15], num_episodes=100, epsilon=epsilon)
    for t in trajectories:
        buffer.add(t[0],t[1])
    average_scores.append(start_score)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        trajectories, cycle_score = collect_trajectories(env, model, [buffer.getAverage()[0],buffer.getMax()], num_episodes=100, epsilon=epsilon)
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
        action_1_accuracy = []
        action_2_accuracy = []
        action_3_accuracy = []
        entropy_a1 = []
        entropy_a2 = []
        entropy_a3 = []
        # print('about to start the learning')
        for i in range(10):
            batch = buffer.sample_batch(batch_size)
            obs_batch = batch['observations']
            action_batch = batch['actions'] 
            logits = model(obs_batch)
            a1loss = criterion(logits['a1'], action_batch[:,0])#/np.log(3)
            a2loss = criterion(logits['a2'], action_batch[:,1])#/np.log(3)
            a3loss = criterion(logits['a3'], action_batch[:,2])#/np.log(22)
            net_loss = a1loss+a2loss+a3loss
            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()
            total_loss += net_loss.item()
            total_1_loss += a1loss.item()
            total_2_loss += a2loss.item()
            total_3_loss += a3loss.item()
            action_1 = torch.argmax(logits['a1'],dim=1)
            action_2 = torch.argmax(logits['a2'],dim=1)
            action_3 = torch.argmax(logits['a3'],dim=1)
            correct_a1 = (action_1 == action_batch[:, 0]).float()
            action_1_accuracy.append(correct_a1.mean().item())
            correct_a2 = (action_2 == action_batch[:, 1]).float()
            action_2_accuracy.append(correct_a2.mean().item())
            correct_a3 = (action_3 == action_batch[:, 2]).float()
            action_3_accuracy.append(correct_a3.mean().item())
            entropy_a1.append(categorical_entropy(action_batch[:, 0], 3))
            entropy_a2.append(categorical_entropy(action_batch[:, 1], 3))
            entropy_a3.append(categorical_entropy(action_batch[:, 2], 22))
        m, st = buffer.getAverage()
        print(f"Avg Loss:    {total_loss / 10:.3f}")
        print(f"Avg A1 Loss: {total_1_loss / 10:.3f}")
        print(f"Avg A2 Loss: {total_2_loss / 10:.3f}")
        print(f"Avg A3 Loss: {total_3_loss / 10:.3f}")
        print(f"Avg A1 Accuracy: {np.mean(action_1_accuracy):.3f}")
        print(f"Avg A2 Accuracy: {np.mean(action_2_accuracy):.3f}")
        print(f"Avg A3 Accuracy: {np.mean(action_3_accuracy):.3f}")
        print(f"Avg A1 Entropy: {np.mean(entropy_a1):.3f}")
        print(f"Avg A2 Entropy: {np.mean(entropy_a2):.3f}")
        print(f"Avg A3 Entropy: {np.mean(entropy_a3):.3f}")
        print('best performing run: ', buffer.getMax())
        print(f'buffer average score: {m:0.3f} +/- {st:0.3f}')
        print(f'buffer size: {len(buffer)}')
        print(f'epsilon: {epsilon:0.3f}')
        average_losses.append(total_loss)
        epsilon -= estep
    return model, {'score':average_scores, 'loss':average_losses}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CalicoEnv()
model = ActionDecoder(128,4)

replay_buffer = TransformerBuffer(4000)

trained_model, info = train_model(env,model,replay_buffer,1000)

plt.plot(range(len(info['score'])),info['score'])
# plt.plot(range(len(info['score'])),info['score'])
plt.show()
torch.save(trained_model.state_dict(), 'trained_calico.pt')