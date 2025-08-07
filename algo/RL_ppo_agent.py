import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import argparse
from network.RL_network import RL_network
import os

class RL_PPOAgent:
    def __init__(self, input_dim, output_dim, lr, gamma, clip_epsilon, num_layers, hidden_dim, device, load=False, checkpoint_path=None):
        self.device = device
        
        # Create model and set precision based on the current gs.init precision
        self.model = RL_network(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        
        # Set tensor precision
        self.dtype = torch.float64
        
        # Convert model parameters to the correct precision
        self.model = self.model.to(dtype=self.dtype)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.checkpoint_path = checkpoint_path
        self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state successfully loaded")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state successfully loaded")
        
        self.model.eval()
        print(f"Checkpoint loaded from {self.checkpoint_path}")

    def select_action(self, RL_state):
        # Ensure state has the right precision
        RL_state = RL_state.to(dtype=self.dtype)
        
        with torch.no_grad():
            logits = self.model(RL_state)
        probs = nn.functional.softmax(logits, dim=-1)
        
        dist = Categorical(probs)
        action = dist.sample()
        return action

    def run(self, RL_states, RL_actions, RL_dones):
        # Convert all inputs to the right precision
        RL_states = torch.stack(RL_states).to(dtype=self.dtype)
        RL_actions = torch.stack(RL_actions).to(dtype=self.dtype)
        RL_dones = torch.stack(RL_dones).to(dtype=torch.bool)
