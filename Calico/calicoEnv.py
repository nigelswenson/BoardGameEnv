from gameComponents import *
from controllers import *
import gymnasium as gym
import torch
from policies import TransformerState
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
            "action_history": gym.spaces.Sequence(gym.spaces.Box(low=0,high=198, shape=(), dtype=np.int32)),
            "return_to_go": gym.spaces.Sequence(gym.spaces.Box(low=0, high=100, shape=(), dtype=np.float32)),
        })
        store_tiles = [t.toList() for t in self.store.openTiles]
        store_tiles = [sublist[0]*6+sublist[1] for sublist in store_tiles]
        inventory_tiles = [t.toList() for t in self.inventory]
        inventory_tiles = [sublist[0]*6+sublist[1] for  sublist in inventory_tiles]
        board_stuff =self.board.grid.toList()
        self.state = TransformerState(board_stuff[0], board_stuff[1], store_tiles,inventory_tiles)
        self.space_key = copy.deepcopy(self.board.openSpaces)
        self.action_mask = np.ones(22)
        self.action_hist = [198]
        self.reward_hist = [10]
        self.training = True

    def _get_obs(self):
        # print('return to go',torch.tensor(self.reward_hist, dtype=torch.float32))
        return {
            "state": self.state,
            "action_history": torch.tensor(self.action_hist, dtype=torch.int32),
            "return_to_go": torch.tensor(self.reward_hist, dtype=torch.float32),
        }
    
    def step(self, action):
        # Action will be 
        # 1. which store piece to grab
        # 2. which hand piece to grab
        # 3. which spot to place it on
        # print(action['a3'])
        action1,action2,action3=action[0],action[1],action[2]
        action_value = action1 * 3*22 + action2*22 + action3 
        action_value= action_value.reshape([1])
        self.action_hist.append(action_value)
        selected = self.store.selectTile(action1)
        self.inventory.append(selected)
        placement_tile = self.inventory.pop(action2)
        self.board.placeTile(placement_tile,self.space_key[action3])
        self.action_mask[action3] = 0
        self.board.checkHexes()
        self.board.checkPins()
        self.board.checkCats()
        # randomly remove a tile from the store
        # print(len(self.store.tileCollection))
        self.store.selectTile(np.random.randint(0,3))
        current_score = self.board.getScore()
        reward = current_score
        self.reward_hist.append(max(self.reward_hist[0]-current_score,0))
        done = len(self.board.openSpaces) <= 0
        store_tiles = [t.toList() for t in self.store.openTiles]
        store_tiles = [sublist[0]*6+sublist[1] for sublist in store_tiles]
        inventory_tiles = [t.toList() for t in self.inventory]
        inventory_tiles = [sublist[0]*6+sublist[1] for  sublist in inventory_tiles]
        board_stuff = self.board.grid.toList()
        state_dict = {'board':board_stuff[0],'neighbors':board_stuff[1],
                      'store':store_tiles,'hand':inventory_tiles}
        self.state.addTstep(state_dict)


        return self._get_obs(), reward, done, False, {'action_mask':self.action_mask}

    def get_action_mask(self):
        return self.action_mask

    def reset(self,desired_return):
        self.board.reset()
        self.store.reset()
        self.inventory = [self.store.tileCollection.pop(),self.store.tileCollection.pop()]
        self.action_mask = np.ones(22)
        self.reward_hist= [desired_return]
        self.action_hist = [198]
        store_tiles = [t.toList() for t in self.store.openTiles]
        store_tiles = [sublist[0]*6+sublist[1] for sublist in store_tiles]
        inventory_tiles = [t.toList() for t in self.inventory]
        inventory_tiles = [sublist[0]*6+sublist[1] for  sublist in inventory_tiles]
        board_stuff = self.board.grid.toList()
        self.state = TransformerState(board_stuff[0], board_stuff[1], store_tiles,inventory_tiles)
        return self._get_obs(), {'action_mask':self.action_mask}
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True