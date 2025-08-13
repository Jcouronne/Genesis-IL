import argparse
import genesis as gs
import torch
from env.pick_place_random_block import PickPlaceRandomBlockEnv
from algo.RL_ppo_agent import RL_PPOAgent  # Expert RL network
from algo.IL_agent import IL_Agent     # IL learner
from env import *
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time

num_episodes = 1000
lr=1e-3
gamma=0.99
clip_epsilon=0.15
num_layers = 12
hidden_dim = 32

gs.init(backend=gs.gpu, precision="64")

task_to_class = {
    'PickPlaceRandomBlock': PickPlaceRandomBlockEnv,
}

def create_environment(task_name):
    if task_name in task_to_class:
        return task_to_class[task_name] 
    else:
        raise ValueError(f"Task '{task_name}' is not recognized.")

def train_imitation_learning(args, lr, num_layers, hidden_dim):
    # Setup checkpoints
    RL_checkpoint_path = f"logs/{args.task}_ppo_checkpoint_released.pth"
    
    if args.load_path is not None:
        load_IL = True
        IL_checkpoint_path = f"logs/{args.task}_IL_checkpoint_released.pth"
        print(f"Loading IL checkpoint from {IL_checkpoint_path}")
    else:
        load_IL = False
        IL_checkpoint_path = f"logs/{args.task}_IL_checkpoint.pth"
        print(f"Creating new IL checkpoint at {IL_checkpoint_path}")
    
    os.makedirs(os.path.dirname(IL_checkpoint_path), exist_ok=True)
    
    # Create environment
    env = create_environment(args.task)(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"Created environment: {env}")

    # Load RL agent
    RL_ppo_agent = RL_PPOAgent(
        input_dim=env.state_dim, 
        output_dim=env.action_space, 
        lr=lr, gamma=gamma, clip_epsilon=clip_epsilon, 
        num_layers=num_layers, hidden_dim=hidden_dim, 
        device=args.device, 
        load=True,  # Load pre-trained RL
        checkpoint_path=RL_checkpoint_path
    )
    
    # Create IL agent - load if -l flag is used
    IL_agent = IL_Agent(
        input_dim=env.state_dim,
        output_dim=env.action_space,
        lr=lr,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        device=args.device,
        load=load_IL,  # Load if -l flag is used
        checkpoint_path=IL_checkpoint_path
    )
    
    if args.device == "mps":
        gs.tools.run_in_another_thread(fn=IL_run, args=(env, RL_ppo_agent, IL_agent, num_episodes))
        env.scene.viewer.start()
    else:
        IL_run(env, RL_ppo_agent, IL_agent, num_episodes)

def IL_run(env, RL_ppo_agent, IL_agent, num_episodes):
    # Training statistics
    episode_stats = []
    IL_loss_stats = []
    RL_rewards_stats = []
    IL_rewards_stats = []
    IL_done_stats = []  # Track IL completion rates
    RL_done_stats = []  # Track RL completion rates
    
    # Expert data collection buffer
    RL_states_buffer = []
    RL_actions_buffer = []
    collect_frequency = 10  # Train IL every N episodes
    
    for episode in range(num_episodes):
        print("Starting episode:", episode + 1)
        # Collect RL demonstration
        RL_state = env.reset()
        RL_done_array = torch.tensor([False] * env.num_envs).to(args.device)
        
        episode_RL_states = []
        episode_RL_actions = []
        RL_total_reward = 0
        
        # Expert generates demonstration
        for step in range(5):
            RL_action = RL_ppo_agent.select_action(RL_state)
            RL_next_state, RL_reward, RL_done = env.step(RL_action)

            # Store RL demonstration
            episode_RL_states.append(RL_state.clone())
            episode_RL_actions.append(RL_action.clone())

            RL_state = RL_next_state
            RL_total_reward += RL_reward.sum().item()
            RL_done_array = torch.logical_or(RL_done_array, RL_done)

            # Early termination if all environments are done
            if RL_done_array.all():
                break

        # Calculate RL completion rate based on actual dones (not reward threshold)
        RL_completion_rate = (RL_done_array.sum().item() / env.num_envs) * 100.0
        
        # Add to buffer
        RL_states_buffer.extend(episode_RL_states)
        RL_actions_buffer.extend(episode_RL_actions)
        
        print(f"Episode {episode}, Expert Reward: {RL_total_reward:.2f}, Buffer Size: {len(RL_states_buffer)}")
        
        # Train IL agent periodically
        if (episode + 1) % collect_frequency == 0 and len(RL_states_buffer) > 0:
            print(f"Training IL agent on {len(RL_states_buffer)} RL samples...")
            
            avg_loss = IL_agent.train_batch(
                RL_states_buffer, 
                RL_actions_buffer, 
                batch_size=64
            )
            
            print(f"IL Training Loss: {avg_loss:.6f}")
            
            # Test IL agent performance
            IL_reward, IL_completion = test_IL_performance(IL_agent, env)
            
            # Store statistics
            episode_stats.append(episode)
            IL_loss_stats.append(avg_loss)
            RL_rewards_stats.append(RL_total_reward)
            IL_rewards_stats.append(IL_reward)
            IL_done_stats.append(IL_completion)
            RL_done_stats.append(RL_completion_rate)
            
            # Clear buffer to manage memory
            RL_states_buffer = RL_states_buffer[-100:]  # Keep recent 100 samples
            RL_actions_buffer = RL_actions_buffer[-100:]
            
            # Save IL checkpoint
            IL_agent.save_checkpoint()
        
        if episode % 50 == 0 or episode == num_episodes - 1:
            # Periodic testing
            print(f"Testing IL agent at episode {episode}...")
            test_IL_performance(IL_agent, env, verbose=True)
    
    # Plot results
    plot_IL_results(episode_stats, IL_loss_stats, RL_rewards_stats, IL_rewards_stats, 
                   IL_done_stats, RL_done_stats, lr, num_layers, hidden_dim)

def test_IL_performance(IL_agent, env, num_test_episodes=3, verbose=False):
    """Test IL agent performance during training"""
    IL_agent.model.eval()
    IL_total_rewards = []
    IL_completion_rates = []
    
    for episode in range(num_test_episodes):
        IL_state = env.reset()
        IL_total_reward = 0
        IL_done_array = torch.tensor([False] * env.num_envs).to(args.device)

        for step in range(5):
            if verbose:
                print(f"Testing IL Agent - Episode {episode + 1}, Step {step + 1}")
            IL_action = IL_agent.select_action(IL_state)
            IL_next_state, IL_reward, IL_done = env.step(IL_action)
            IL_total_reward += IL_reward.sum().item()
            IL_state = IL_next_state
            IL_done_array = torch.logical_or(IL_done_array, IL_done)

            if IL_done_array.all():
                break

        IL_total_rewards.append(IL_total_reward)
        # Calculate completion rate based on actual dones (matching environment pattern)
        completion_rate = (IL_done_array.sum().item() / env.num_envs) * 100.0
        IL_completion_rates.append(completion_rate)

    IL_avg_reward = sum(IL_total_rewards) / len(IL_total_rewards)
    IL_avg_completion = sum(IL_completion_rates) / len(IL_completion_rates)
    
    if verbose:
        print(f"IL Agent Average Test Reward: {IL_avg_reward:.4f}")
        print(f"IL Agent Average Completion Rate: {IL_avg_completion:.2f}%")

    return IL_avg_reward, IL_avg_completion

def plot_IL_results(episodes, losses, RL_rewards, IL_rewards, IL_done_rates, RL_done_rates, lr, num_layers, hidden_dim):
    """Plot IL training results"""
    figure, axis = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot IL training loss
    axis[0,0].plot(episodes, losses, color='b', label='IL Loss')
    axis[0,0].set_title('IL Training Loss')
    axis[0,0].set_xlabel('Episode')
    axis[0,0].set_ylabel('Loss')
    axis[0,0].legend()
    
    # Plot RL vs IL rewards
    axis[0,1].plot(episodes, RL_rewards, color='g', label='Expert Reward')
    axis[0,1].plot(episodes, IL_rewards, color='r', label='IL Reward')
    axis[0,1].set_title('Expert vs IL Performance')
    axis[0,1].set_xlabel('Episode')
    axis[0,1].set_ylabel('Reward')
    axis[0,1].legend()
    
    # Plot reward difference
    reward_diff = [RL - IL for RL, IL in zip(RL_rewards, IL_rewards)]
    axis[1,0].plot(episodes, reward_diff, color='orange', label='Performance Gap')
    axis[1,0].set_title('Expert-IL Performance Gap')
    axis[1,0].set_xlabel('Episode')
    axis[1,0].set_ylabel('Reward Difference')
    axis[1,0].legend()
    
    # Plot completion rates (both RL and IL)
    if len(IL_done_rates) > 0 or len(RL_done_rates) > 0:
        if len(RL_done_rates) > 0:
            axis[1,1].plot(episodes, RL_done_rates, color='green', label='Expert (RL) Completion Rate')
        if len(IL_done_rates) > 0:
            axis[1,1].plot(episodes, IL_done_rates, color='purple', label='IL Completion Rate')
        axis[1,1].set_title('Task Completion Rates Comparison')
        axis[1,1].set_xlabel('Episode')
        axis[1,1].set_ylabel('Completion Rate (%)')
        axis[1,1].set_ylim(0, 100)
        axis[1,1].grid(True, alpha=0.3)
        axis[1,1].legend()
    
    plt.suptitle(f"Imitation Learning - LR: {lr}, Layers: {num_layers}, Hidden: {hidden_dim}")
    
    ts = time.time()
    timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    os.makedirs("/home/devtex/Documents/Genesis/graphs", exist_ok=True)
    plt.savefig(f"/home/devtex/Documents/Genesis/graphs/IL_{timestamp}.png")
    plt.show()

def test_IL_model(args, lr, num_layers, hidden_dim):
    """Test mode for IL model - evaluates performance without training"""
    print("=" * 60)
    print("IMITATION LEARNING MODEL TEST MODE")
    print("=" * 60)
    
    # Setup checkpoint path - use _released for evaluation
    IL_checkpoint_path = f"logs/{args.task}_IL_checkpoint_released.pth"
    
    if not os.path.exists(IL_checkpoint_path):
        print(f"Error: IL checkpoint not found at {IL_checkpoint_path}")
        print("Please train the IL model first or check the checkpoint path.")
        return
    
    # Create environment for testing
    env = create_environment(args.task)(vis=args.vis, device=args.device, num_envs=args.num_envs)
    print(f"Created test environment: {env}")
    
    # Load IL agent from checkpoint
    IL_agent = IL_Agent(
        input_dim=env.state_dim,
        output_dim=env.action_space,
        lr=lr,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        device=args.device,
        load=True,  # Load trained IL model
        checkpoint_path=IL_checkpoint_path
    )
    
    print(f"Loaded IL agent from: {IL_checkpoint_path}")
    
    # Run simple test
    run_IL_test_suite(env, IL_agent, args)

def run_IL_test_suite(env, IL_agent, args):
    """Run test suite for IL model - 50 episodes with reward and completion tracking"""
    print("\n" + "-" * 40)
    print("Running IL Test - 50 Episodes")
    print("-" * 40)
    
    # Simple test: 50 episodes
    episode_rewards, episode_completions = test_IL_simple_performance(IL_agent, env, args, num_episodes=50)

    # Plot test results
    plot_test_results(episode_rewards, episode_completions, args)
    
    # Simple report
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_completion = sum(episode_completions) / len(episode_completions)
    
    print(f"\nSimple Test Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Completion Rate: {avg_completion:.1f}%")
    print(f"  Total Episodes: {len(episode_rewards)}")

def test_IL_simple_performance(IL_agent, env, args, num_episodes=50):
    """Simple test IL agent performance over episodes"""
    IL_agent.model.eval()
    episode_rewards = []
    episode_completions = []
    
    print("   Running simple performance test...")
    for episode in range(num_episodes):
        IL_state = env.reset()
        IL_total_reward = 0
        IL_done_array = torch.tensor([False] * env.num_envs).to(args.device)
        
        for step in range(5):
            IL_action = IL_agent.select_action(IL_state)
            IL_next_state, IL_reward, IL_done = env.step(IL_action)
            IL_total_reward += IL_reward.sum().item()
            IL_state = IL_next_state
            IL_done_array = torch.logical_or(IL_done_array, IL_done)
            print(IL_done_array)
            if IL_done_array.all():
                break
        
        episode_rewards.append(IL_total_reward)
        # Calculate completion rate based on actual dones (matching environment pattern)
        completion_rate = (IL_done_array.sum().item() / env.num_envs) * 100.0
        episode_completions.append(completion_rate)
        
        if (episode + 1) % 10 == 0:
            print(f"   Episode {episode+1}: Reward={IL_total_reward:.2f}, Completion={completion_rate:.1f}%")
    
    return episode_rewards, episode_completions

def plot_test_results(IL_rewards, IL_dones, args):
    """Plot test results"""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    episodes = range(1, len(IL_rewards)+1)

    # Plot 1: Episode Rewards
    axes[0].plot(episodes, IL_rewards, color='blue', alpha=0.7, linewidth=1.5)
    axes[0].axhline(y=sum(IL_rewards)/len(IL_rewards), color='red', linestyle='--', label=f'Average: {sum(IL_rewards)/len(IL_rewards):.1f}')
    axes[0].set_title('IL Agent - Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Completion Rates
    axes[1].plot(episodes, IL_dones, color='green', alpha=0.7, linewidth=1.5)
    axes[1].axhline(y=sum(IL_dones)/len(IL_dones), color='red', linestyle='--', label=f'Average: {sum(IL_dones)/len(IL_dones):.1f}%')
    axes[1].set_title('IL Agent - Episode Completion Rates')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Completion Rate (%)')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Simple IL Test Results - Task: {args.task}")
    plt.tight_layout()
    
    # Save the plot
    ts = time.time()
    timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    os.makedirs("/home/devtex/Documents/Genesis/graphs", exist_ok=True)
    plt.savefig(f"/home/devtex/Documents/Genesis/graphs/IL_simple_test_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: /home/devtex/Documents/Genesis/graphs/IL_simple_test_{timestamp}.png")
    plt.show()

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Enable visualization") 
    parser.add_argument("-l", "--load_path", action="store_const", const="default", default=None, help="Load IL model from default checkpoint path") 
    parser.add_argument("-e", "--evaluate", action="store_true", default=False, help="Evaluate IL model (test mode)")
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of environments to create") 
    parser.add_argument("-t", "--task", type=str, default="PickPlaceRandomBlock", help="Task to train/test on")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device: cpu or cuda:x or mps for macos")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    
    # Check if we should run in test mode
    if args.evaluate:
        print("Evaluate flag detected - switching to test mode")
        test_IL_model(args, lr, num_layers, hidden_dim)
    else:
        print("Starting IL training mode")
        train_imitation_learning(args, lr, num_layers, hidden_dim)
