import torch
import numpy as np
from torch import nn

class HexConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.center = nn.Linear(in_dim, out_dim)
        self.directions = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(6)])

    def forward(self, x, neighbors):
        """
        x: [B, N, D] node features for each hex cell
        neighbors: [B, N, 6] indices of 6 neighbors for each cell
        """
        center_out = self.center(x)
        agg = center_out
        # print(x.shape, neighbors.shape)
        batched = len(x.shape)==4
        for i, layer in enumerate(self.directions):
            # Gather the i-th neighbor's feature vector
            # neighbor_feats = torch.gather(x, dim=1, index=neighbors[:, :, i].unsqueeze(-1).expand(-1, -1, x.size(-1)))
            if batched:
                agg += layer(neighbors[:,:,:,i])
            else:
                agg += layer(neighbors[:,:,i])
        # print(agg.shape)
        
        return agg

class BoardEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, state_shape):
        super().__init__()
        self.conv1 = HexConv(input_dim, hidden_dim)
        self.finisher = nn.Linear(hidden_dim*state_shape,output_dim)
        self.conv2 = HexConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, neighbors):
        # print(x.shape, neighbors.shape)
        x = self.relu(self.conv1(x, neighbors))
        # x = self.conv2(x, neighbors)
        if len(x.shape) == 3:
            x = self.finisher(x.reshape([x.shape[0], 54*4]))
        else:
            x = self.finisher(x.reshape([x.shape[0], x.shape[1], 54*4]))
        return x

class ActionDecoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.board_state_embedding = BoardEncoder(2,4,int(d_model/2),54)
        self.action_embedding = nn.Embedding(3*3*22 + 1, int(d_model/8))
        self.store_embedding = nn.Embedding(6*6, int(d_model/8))
        self.hand_embedding = nn.Embedding(6*6, int(d_model/8))
        self.reward_embedding = nn.Linear(1,int(d_model/8))
        self.pos_emb = nn.Parameter(torch.randn(2, d_model))
        self.store_combiner = nn.Linear(3,1)
        self.hand_combiner = nn.Linear(2,1)
        self.transformer = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model, num_heads),
            num_layers=2
        )

        # Output heads for each action
        self.heads = nn.ModuleDict({
            "a1": nn.Linear(d_model, 3),
            "a2": nn.Linear(d_model, 3),
            "a3": nn.Linear(d_model, 22),
        })

    def forward(self, input_tokens, action_masks=None):
        """
        input_tokens: dict with keys ['state','action']
                      each tensor is shape [B, 1]
        action_masks: list of shape [B, 22]
                      binary mask where 1 = valid, 0 = invalid
        """
        # B = input_tokens['state'].size(0)
        # Embed all input tokens and stack

        # print('action val', action_value)
        # print(input_tokens['return_to_go'])
        
        board_thing, neighbor_thing, store_thing, hand_thing = input_tokens['state'].getTensor()
        batched = len(neighbor_thing.shape) == 5
        board_part = self.board_state_embedding(board_thing, neighbor_thing)
        if batched:
            store_embedded = self.store_embedding(store_thing).transpose(2,3)
            hand_embedded = self.hand_embedding(hand_thing).transpose(2,3)
            store_part = self.store_combiner(store_embedded)
            store_part = store_part.reshape([board_part.shape[0],board_part.shape[1],16])
            hand_part = self.hand_combiner(hand_embedded)
            hand_part = store_part.reshape([board_part.shape[0],board_part.shape[1],16])
            reward_part = self.reward_embedding(input_tokens['return_to_go'].reshape([input_tokens['return_to_go'].shape[0],input_tokens['return_to_go'].shape[1],1]))
            action_part = self.action_embedding(input_tokens['action_history'])
        else:
            store_embedded = self.store_embedding(store_thing).transpose(1,2)
            hand_embedded = self.hand_embedding(hand_thing).transpose(1,2)
            store_part = self.store_combiner(store_embedded)
            store_part = store_part.reshape([len(board_part),16])
            hand_part = self.hand_combiner(hand_embedded)
            hand_part = store_part.reshape([len(hand_part),16])
            reward_part = self.reward_embedding(input_tokens['return_to_go'].reshape([len(board_part),1]))
            action_part = self.action_embedding(input_tokens['action_history'])#.reshape([len(input_tokens['action_history'])]))
        
        tokens = torch.cat([board_part, store_part, hand_part, action_part,reward_part], dim=-1)  # [B, seq_len, D]
        # print(tokens.shape)
        # print(self.pos_emb.shape)
        # tokens += self.pos_emb #.unsqueeze(0)  # add positional encoding

        # Transpose for transformer [seq_len, B, D]
        # tokens = tokens.transpose(0, 1)

        # Dummy memory (could be full encoder for full Decision Transformer)
        memory = torch.zeros_like(tokens)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(1).to(tokens.device)
        out = self.transformer(tgt=tokens, memory=memory)#, tgt_mask=tgt_mask)  # [seq_len, B, D]
        if batched:
            out = out.transpose(0, 1)  # [B, seq_len, D]
        logits = {
            "a1": self.heads["a1"](out[-1]),  # position of a1
            "a2": self.heads["a2"](out[-1]),
            "a3": self.heads["a3"](out[-1]),
        }
        # Apply action masking
        if action_masks is not None:
            logits['a3'][action_masks == 0] = float("-inf")
        return logits  # raw logits


class TransformerBuffer():
    def __init__(self,buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.max = 0

    def add(self, episode, score):
        self.buffer.append((score, episode))
        self.max = max(score,self.max)
        if len(self.buffer) > self.buffer_size:
            # Keep top-k episodes
            self.buffer.sort(key=lambda x: x[0], reverse=True)
            self.buffer = self.buffer[:self.buffer_size]

    def sample_batch(self, batch_size):
        scores, episodes = zip(*self.buffer)
        things = np.random.choice(episodes, batch_size, replace=False)
        prepped = {'observations':{'state':None,'return_to_go':[], 'action_history':[]},
                  'actions':[]}
                #   'action_mask':[],
                #   'returns':[]}
        index = np.random.randint(22)
        # print('RANDOM INDEX', index)
        state_list = []
        for episode in things:
            state_list.append(episode['observations']['state'])
            prepped['actions'].append(episode['actions'][index])
            # prepped['action_mask'].append(episode['action_mask'][index].tolist())
            # prepped['returns'].append(episode['returns'][index])
            prepped['observations']['action_history'].append(episode['observations']['action_history'][:index+1])
            prepped['observations']['return_to_go'].append(episode['observations']['return_to_go'][:index+1])
        prepped['observations']['state'] = TransformerBatch(state_list, index)
        # print(prepped['observations']['action_history'][0].shape)
        # prepped['action_mask'] = torch.tensor(prepped['action_mask'])
        prepped['actions'] = torch.tensor(prepped['actions'])
        # print(prepped['observations']['return_to_go'][0])
        prepped['observations']['action_history'] = torch.stack(prepped['observations']['action_history'])
        prepped['observations']['return_to_go'] = torch.stack(prepped['observations']['return_to_go'])
        # print(prepped['observations']['action_history'].shape)
        # print(prepped['observations']['return_to_go'].shape)
        return prepped
    
    def getMax(self):
        return self.max
    
    def getAverage(self):
        scores = [a[0] for a in self.buffer]
        return np.average(scores), np.std(scores)
    
    def __len__(self):
        return len(self.buffer)  
class TransformerState():
    def __init__(self,board,neighbors, store,hand):
        self.state =  {'board':[board],'neighbors':[neighbors],'store':[store],'hand':[hand]}

    def addTstep(self,state_dict):
        for key in state_dict.keys():
            self.state[key].append(state_dict[key])
    
    def getTensor(self):
        return torch.tensor(self.state['board'],dtype=torch.float32),torch.tensor(self.state['neighbors'],dtype=torch.float32),torch.tensor(self.state['store'],dtype=torch.int32),torch.tensor(self.state['hand'],dtype=torch.int32)

    def getList(self,index=-1):
        return self.state['board'][:index],self.state['neighbors'][:index],self.state['store'][:index],self.state['hand'][:index]
    
class TransformerBatch():
    def __init__(self, statelist, index):
        self.state =  {'board':[],'neighbors':[],'store':[],'hand':[]}
        for state in statelist:
            self.state['board'].append(state.state['board'][:index+1])
            self.state['neighbors'].append(state.state['neighbors'][:index+1])
            self.state['store'].append(state.state['store'][:index+1])
            self.state['hand'].append(state.state['hand'][:index+1])

    def getTensor(self):
        return torch.tensor(self.state['board'],dtype=torch.float32),torch.tensor(self.state['neighbors'],dtype=torch.float32),torch.tensor(self.state['store'],dtype=torch.int32),torch.tensor(self.state['hand'],dtype=torch.int32)

