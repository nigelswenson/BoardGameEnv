from gameComponents import *
from controllers import *
import gymnasium as gym
import torch
class CalicoEnv(gym.Env):
    def __init__(self, num_agents=1):
        self.store = Store()
        with open('CalicoHexset.json') as jsonFile:
            temp = json.load(jsonFile)
        selections = np.random.choice(list(temp.keys()),3,replace=False)
        self.hexes = []
        for key in selections:
            self.hexes.append(hexset_key[key](temp[key]))

        with open('CalicoReducedCatset.json') as jsonFile:
            temp = json.load(jsonFile)
        selections = np.random.choice(list(temp.values()),3,replace=False)
        orders = list(range(6))
        np.random.shuffle(orders)
        self.catsUsed = [CatClusterManager(sel['size'],orders[i*2:i*2+2],sel['points'])
                          for i,sel in enumerate(selections)]
        
        self.inventory = [self.store.tileCollection.pop(),self.store.tileCollection.pop()]
        with open('CalicoBoardset1.json') as jsonFile:
            boardInfo = json.load(jsonFile)
        self.board = Board(boardInfo, {'cats':self.catsUsed,'hexes':self.hexes})

        self.action_space = gym.spaces.MultiDiscrete([3,3,22]) 
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            "action_history": gym.spaces.Sequence(gym.spaces.MultiDiscrete([3,3,22])),
            "reward_history": gym.spaces.Sequence(gym.spaces.Box(low=0, high=1, shape=(), dtype=np.float32)),
        })
        store_tiles = [t.toList() for t in self.store.openTiles]
        inventory_tiles = [t.toList() for t in self.inventory]
        self.state = {'board':self.board.grid.toTensor(),'store':torch.tensor(store_tiles),'hand':torch.tensor(inventory_tiles)}
        self.space_key = copy.deepcopy(self.board.openSpaces)
        self.action_mask = np.ones(22)
        self.action_hist = []
        self.reward_hist = [0]
        self.training = True
    def _get_obs(self):
        return {
            "state": self.state,
            "action_history": torch.tensor(self.action_hist, dtype=torch.int32),
            "reward_history": torch.tensor(self.reward_hist, dtype=torch.float32),
        }
    
    def step(self, action):
        # Action will be 
        # 1. which store piece to grab
        # 2. which hand piece to grab
        # 3. which spot to place it on
        # print(action['a3'])
        if self.training:
            distribution = torch.distributions.Categorical(logits=action['a1'])
            action1 = distribution.sample()
            distribution = torch.distributions.Categorical(logits=action['a2'])
            action2 = distribution.sample()
            distribution = torch.distributions.Categorical(logits=action['a3'])
            action3 = distribution.sample()
        else:
            action1 = torch.argmax(action['a1'])
            action2 = torch.argmax(action['a2'])
            action3 = torch.argmax(action['a3'])
        # print(action1,action2,action3)
        self.action_hist.append([action1,action2,action3])
        selected = self.store.selectTile(action1)
        self.inventory.append(selected)
        placement_tile = self.inventory.pop(action2)
        self.board.placeTile(placement_tile,self.space_key[action3])
        self.action_mask[action3] = 0
        self.board.checkHexes()
        self.board.checkPins()
        self.board.checkCats()
        current_score = self.board.getScore()
        reward = current_score - self.reward_hist[-1]
        self.reward_hist.append(reward)
        done = len(self.board.openSpaces) <= 0
        store_tiles = [t.toList() for t in self.store.openTiles]
        inventory_tiles = [t.toList() for t in self.inventory]
        self.state = {'board':self.board.grid.toTensor(),'store':torch.tensor(store_tiles),'hand':torch.tensor(inventory_tiles)}
        return self._get_obs(), reward, done, False, {'action_mask':self.action_mask}

    def get_action_mask(self):
        return self.action_mask

    def reset(self):
        self.board.reset()
        self.store.reset()
        self.inventory = [self.store.tileCollection.pop(),self.store.tileCollection.pop()]
        self.action_mask = np.ones(22)
        self.reward_hist= [0]
        self.action_hist = []
        return self._get_obs(), {'action_mask':self.action_mask}