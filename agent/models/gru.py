# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h=None):
        # x shape: (batch_size, seq_len, state_size)
        # h shape: (1, batch_size, hidden_size)
        if not h:
            h = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        x = F.relu(self.fc1(x.view(x.size(0)*x.size(1), -1))).view(x.size(0), x.size(1), -1)
        x, _ = self.fc2(x, h)
        o = F.softmax(self.fc3(x[:, -1, :]), dim=-1)
        return o    # shape: (batch_size, 1)
