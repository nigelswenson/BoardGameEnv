import torch

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
        for i, layer in enumerate(self.directions):
            # Gather the i-th neighbor's feature vector
            # neighbor_feats = torch.gather(x, dim=1, index=neighbors[:, :, i].unsqueeze(-1).expand(-1, -1, x.size(-1)))
            agg += layer(neighbors[:,i])
        return agg

class BoardEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = HexConv(input_dim, hidden_dim)
        self.conv2 = HexConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, neighbors):
        # print(x.shape)
        # x = self.relu(self.conv1(x, neighbors))
        # print(x.shape)
        # x = self.conv2(x, neighbors)
        x = self.conv1(x,neighbors)
        return x

class ActionDecoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.board_state_embedding = BoardEncoder(2,128,8)
        self.action_embedding = nn.Embedding(3*3*22 + 1, d_model)
        self.store_embedding = nn.Embedding(6*6*3, d_model)
        self.hand_embedding = nn.Embedding(6*6*3, d_model)
        self.pos_emb = nn.Parameter(torch.randn(2, d_model))

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
        if len(input_tokens['action_history']) > 0:
            # print(input_tokens['action_history'])
            action_value = input_tokens['action_history'][0][0] * 3*22 + input_tokens['action_history'][0][1]*22 + input_tokens['action_history'][0][2] 
            action_value= action_value.reshape([1])
        else:
            action_value = torch.tensor([198])
        # print('action val', action_value)
        board_part = self.board_state_embedding(input_tokens['state']['board'][0],input_tokens['state']['board'][1])
        store_part = self.store_embedding(input_tokens['state']['store'].reshape(-1))
        hand_part = self.hand_embedding(input_tokens['state']['hand'].reshape(-1))
        action_part = self.action_embedding(action_value)
        # print(board_part.shape)
        # print(store_part.shape)
        # print(hand_part.shape)
        # print(action_part.shape)
        tokens = torch.cat([board_part, store_part, hand_part, action_part], dim=0)  # [B, seq_len, D]
        # print(tokens.shape)
        # print(self.pos_emb.shape)
        # tokens += self.pos_emb #.unsqueeze(0)  # add positional encoding

        # Transpose for transformer [seq_len, B, D]
        # tokens = tokens.transpose(0, 1)

        # Dummy memory (could be full encoder for full Decision Transformer)
        memory = torch.zeros_like(tokens)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(65).to(tokens.device)
        out = self.transformer(tgt=tokens, memory=memory, tgt_mask=tgt_mask)  # [seq_len, B, D]

        out = out.transpose(0, 1)  # [B, seq_len, D]

        logits = {
            "a1": self.heads["a1"](out[:, 2]),  # position of a1
            "a2": self.heads["a2"](out[:, 3]),
            "a3": self.heads["a3"](out[:, 4]),
        }
        # Apply action masking
        if action_masks is not None:
            logits['a3'][action_masks == 0] = float("-inf")
        return logits  # raw logits
