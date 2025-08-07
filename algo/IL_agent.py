import torch
import torch.nn as nn
import torch.optim
from torch.distributions import Categorical
from network.IL_network import IL_network
import os

class IL_Agent:
    def __init__(self, input_dim, output_dim, lr, num_layers, hidden_dim, device, load=False, checkpoint_path=None):
        self.device = device
        
        # Create model and set precision based on the current gs.init precision
        self.model = IL_network(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        self.dtype = torch.float64 # Tensor precision
        
        # Convert model parameters to the correct precision
        self.model = self.model.to(dtype=self.dtype)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_path = checkpoint_path
        
        if load and os.path.exists(checkpoint_path):
            self.load_checkpoint()

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"IL Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state successfully loaded")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state successfully loaded")
        
        self.model.eval()  # Set to evaluation mode
        print(f"IL Checkpoint loaded from {self.checkpoint_path}")

    def select_action(self, RL_state):
        # Ensure state has the right precision
        RL_state = RL_state.to(dtype=self.dtype)

        with torch.no_grad():
            logits = self.model(RL_state)
        probs = nn.functional.softmax(logits, dim=-1)
        dist = Categorical(probs)
        IL_action = dist.sample()
        return IL_action

    def train(self, RL_states, RL_actions, dones):
        self.model.train()  # Set to training mode
    
        # Convert all inputs to the right precision and device
        RL_states = torch.stack(RL_states).to(dtype=self.dtype, device=self.device)
        RL_actions = torch.stack(RL_actions).to(dtype=torch.long, device=self.device)
    
        # Forward pass to get logits (not actions)
        logits = self.model(RL_states)
    
        # Use CrossEntropyLoss for discrete action classification
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, RL_actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        self.model.eval()  # Set back to eval mode
        return loss.item()

    def train_batch(self, RL_states, RL_actions, batch_size=32):
        """Train on a batch of expert data"""
        self.model.train()
        
        # Convert list of tensors to single tensor and flatten multi-env data
        states_tensor = torch.stack(RL_states).to(dtype=self.dtype, device=self.device)
        actions_tensor = torch.stack(RL_actions).to(dtype=torch.long, device=self.device)
        
        # Flatten if multi-environment: [steps, envs, ...] -> [steps*envs, ...]
        if states_tensor.dim() > 2:
            batch_size_total = states_tensor.shape[0] * states_tensor.shape[1]
            states_flat = states_tensor.view(batch_size_total, -1)
            actions_flat = actions_tensor.view(batch_size_total)
        else:
            states_flat = states_tensor
            actions_flat = actions_tensor
        
        print(f"Flattened shapes - States: {states_flat.shape}, Actions: {actions_flat.shape}")
        
        # Train on flattened data
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(states_flat), batch_size):
            batch_states = states_flat[i:i+batch_size]
            batch_actions = actions_flat[i:i+batch_size]
            
            logits = self.model(batch_states)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_actions)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.eval()
        return total_loss / num_batches if num_batches > 0 else 0
