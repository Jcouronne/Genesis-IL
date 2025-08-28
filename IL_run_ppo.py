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
import numpy as np

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
    # Add start time tracking
    start_time = time.time()
    
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
    
    # Variance tracking
    window_size = 10
    
    # Setup interactive plotting
    plt.ion()
    figure, axis = plt.subplots(2, 2, figsize=(15, 10))
    
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
            
            # Update plots with runtime and variance
            if len(episode_stats) > 1:
                # Calculate current running time
                current_time = time.time()
                elapsed_time = current_time - start_time
                elapsed_minutes = int(elapsed_time // 60)
                elapsed_seconds = int(elapsed_time % 60)
                
                update_IL_plots(axis, episode_stats, IL_loss_stats, RL_rewards_stats, 
                               IL_rewards_stats, IL_done_stats, RL_done_stats, 
                               lr, num_layers, hidden_dim, elapsed_minutes, elapsed_seconds)
        
        if episode % 50 == 0 or episode == num_episodes - 1:
            # Periodic testing
            print(f"Testing IL agent at episode {episode}...")
            test_IL_performance(IL_agent, env, verbose=True)
    
    # Calculate final running time
    final_time = time.time()
    total_elapsed = final_time - start_time
    total_minutes = int(total_elapsed // 60)
    total_seconds = int(total_elapsed % 60)
    
    # Turn off interactive mode and plot final results
    plt.ioff()
    plot_IL_results(episode_stats, IL_loss_stats, RL_rewards_stats, IL_rewards_stats, 
                   IL_done_stats, RL_done_stats, lr, num_layers, hidden_dim, 
                   total_minutes, total_seconds)

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

def update_IL_plots(axis, episodes, losses, RL_rewards, IL_rewards, IL_done_rates, RL_done_rates,
                    lr, num_layers, hidden_dim, elapsed_minutes, elapsed_seconds):
    """Update real-time IL training plots with variance bands"""
    
    # Clear all subplots
    for ax in axis.flat:
        ax.clear()
    
    # Convert to numpy for easier manipulation
    episodes_np = np.array(episodes)
    losses_np = np.array(losses)
    RL_rewards_np = np.array(RL_rewards)
    IL_rewards_np = np.array(IL_rewards)
    IL_done_rates_np = np.array(IL_done_rates)
    RL_done_rates_np = np.array(RL_done_rates)
    
    # Calculate variance bands
    window_size = min(5, len(episodes))  # Adaptive window size
    
    def calculate_std_bands(data_list, window_size):
        std_bands = []
        for i in range(len(data_list)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(data_list), i + window_size//2 + 1)
            
            if end_idx - start_idx > 1:
                window_data = data_list[start_idx:end_idx]
                std_bands.append(np.std(window_data))
            else:
                std_bands.append(0)
        return np.array(std_bands)
    
    # Calculate std bands for all metrics
    losses_std = calculate_std_bands(losses, window_size)
    RL_rewards_std = calculate_std_bands(RL_rewards, window_size)
    IL_rewards_std = calculate_std_bands(IL_rewards, window_size)
    IL_done_std = calculate_std_bands(IL_done_rates, window_size)
    
    # Plot 1: IL training loss with variance
    axis[0,0].plot(episodes_np, losses_np, color='b', linewidth=2, label='IL Loss')
    axis[0,0].fill_between(episodes_np, 
                          losses_np - losses_std, 
                          losses_np + losses_std, 
                          color='b', alpha=0.3, label='±1 STD')
    axis[0,0].set_title('IL Training Loss')
    axis[0,0].set_xlabel('Episode')
    axis[0,0].set_ylabel('Loss')
    axis[0,0].legend()
    axis[0,0].grid(True, alpha=0.3)
    
    # Plot 2: RL vs IL rewards with variance
    axis[0,1].plot(episodes_np, RL_rewards_np, color='g', linewidth=2, label='Expert Reward')
    axis[0,1].fill_between(episodes_np, 
                          RL_rewards_np - RL_rewards_std, 
                          RL_rewards_np + RL_rewards_std, 
                          color='g', alpha=0.3)
    axis[0,1].plot(episodes_np, IL_rewards_np, color='r', linewidth=2, label='IL Reward')
    axis[0,1].fill_between(episodes_np, 
                          IL_rewards_np - IL_rewards_std, 
                          IL_rewards_np + IL_rewards_std, 
                          color='r', alpha=0.3)
    axis[0,1].set_title('Expert vs IL Performance')
    axis[0,1].set_xlabel('Episode')
    axis[0,1].set_ylabel('Reward')
    axis[0,1].legend()
    axis[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Performance gap with variance
    reward_diff = [(RL - IL)**2 for RL, IL in zip(RL_rewards, IL_rewards)]
    reward_diff_np = np.array(reward_diff)
    diff_std = calculate_std_bands(reward_diff, window_size)
    
    axis[1,0].plot(episodes_np, reward_diff_np, color='orange', linewidth=2, label='Performance Gap')
    axis[1,0].fill_between(episodes_np, 
                          reward_diff_np - diff_std, 
                          reward_diff_np + diff_std, 
                          color='orange', alpha=0.3, label='±1 STD')
    axis[1,0].set_title('Expert-IL Performance Gap')
    axis[1,0].set_xlabel('Episode')
    axis[1,0].set_ylabel('Reward Difference²')
    axis[1,0].legend()
    axis[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Completion rates with variance
    if len(RL_done_rates) > 0:
        RL_done_std = calculate_std_bands(RL_done_rates, window_size)
        axis[1,1].plot(episodes_np, RL_done_rates_np, color='green', linewidth=2, label='Expert (RL) Completion Rate')
        axis[1,1].fill_between(episodes_np, 
                              RL_done_rates_np - RL_done_std, 
                              RL_done_rates_np + RL_done_std, 
                              color='green', alpha=0.3)
    
    if len(IL_done_rates) > 0:
        axis[1,1].plot(episodes_np, IL_done_rates_np, color='purple', linewidth=2, label='IL Completion Rate')
        axis[1,1].fill_between(episodes_np, 
                              IL_done_rates_np - IL_done_std, 
                              IL_done_rates_np + IL_done_std, 
                              color='purple', alpha=0.3)
    
    axis[1,1].set_title('Task Completion Rates Comparison')
    axis[1,1].set_xlabel('Episode')
    axis[1,1].set_ylabel('Completion Rate (%)')
    axis[1,1].set_ylim(0, 100)
    axis[1,1].grid(True, alpha=0.3)
    axis[1,1].legend()
    
    plt.suptitle(f"Runtime: {elapsed_minutes}m {elapsed_seconds}s - Imitation Learning - LR: {lr}, Layers: {num_layers}, Hidden: {hidden_dim}")
    plt.tight_layout()
    plt.pause(0.01)

def plot_IL_results(episodes, losses, RL_rewards, IL_rewards, IL_done_rates, RL_done_rates, 
                   lr, num_layers, hidden_dim, total_minutes=None, total_seconds=None):
    """Plot final IL training results with variance bands"""
    figure = plt.figure(figsize=(15, 10))
    
    # Convert to numpy for easier manipulation
    episodes_np = np.array(episodes)
    losses_np = np.array(losses)
    RL_rewards_np = np.array(RL_rewards)
    IL_rewards_np = np.array(IL_rewards)
    IL_done_rates_np = np.array(IL_done_rates)
    RL_done_rates_np = np.array(RL_done_rates)
    
    # Calculate final variance bands
    window_size = min(10, len(episodes))
    
    def calculate_final_std_bands(data_list, window_size):
        std_bands = []
        for i in range(len(data_list)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(data_list), i + window_size//2 + 1)
            
            if end_idx - start_idx > 1:
                window_data = data_list[start_idx:end_idx]
                std_bands.append(np.std(window_data))
            else:
                std_bands.append(0)
        return np.array(std_bands)
    
    # Calculate final std bands
    final_losses_std = calculate_final_std_bands(losses, window_size)
    final_RL_rewards_std = calculate_final_std_bands(RL_rewards, window_size)
    final_IL_rewards_std = calculate_final_std_bands(IL_rewards, window_size)
    final_IL_done_std = calculate_final_std_bands(IL_done_rates, window_size)
    
    # Plot 1: Final IL training loss with variance
    plt.subplot(2, 2, 1)
    plt.plot(episodes_np, losses_np, color='b', linewidth=2, label='IL Loss')
    plt.fill_between(episodes_np, 
                          losses_np - final_losses_std, 
                          losses_np + final_losses_std, 
                          color='b', alpha=0.3, label='±1 STD')
    plt.title('Final IL Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Final RL vs IL rewards with variance
    plt.subplot(2, 2, 2)
    plt.plot(episodes_np, RL_rewards_np, color='g', linewidth=2, label='Expert Reward')
    plt.fill_between(episodes_np, 
                          RL_rewards_np - final_RL_rewards_std, 
                          RL_rewards_np + final_RL_rewards_std, 
                          color='g', alpha=0.3)
    plt.plot(episodes_np, IL_rewards_np, color='r', linewidth=2, label='IL Reward')
    plt.fill_between(episodes_np, 
                          IL_rewards_np - final_IL_rewards_std, 
                          IL_rewards_np + final_IL_rewards_std, 
                          color='r', alpha=0.3)
    plt.title('Final Expert vs IL Performance')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final performance gap with variance
    plt.subplot(2, 2, 3)
    reward_diff = [(RL - IL)**2 for RL, IL in zip(RL_rewards, IL_rewards)]
    reward_diff_np = np.array(reward_diff)
    final_diff_std = calculate_final_std_bands(reward_diff, window_size)
    
    plt.plot(episodes_np, reward_diff_np, color='orange', linewidth=2, label='Performance Gap')
    plt.fill_between(episodes_np, 
                          reward_diff_np - final_diff_std, 
                          reward_diff_np + final_diff_std, 
                          color='orange', alpha=0.3, label='±1 STD')
    plt.title('Final Expert-IL Performance Gap')
    plt.xlabel('Episode')
    plt.ylabel('Reward Difference²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Final completion rates with variance
    plt.subplot(2, 2, 4)
    if len(RL_done_rates) > 0:
        final_RL_done_std = calculate_final_std_bands(RL_done_rates, window_size)
        plt.plot(episodes_np, RL_done_rates_np, color='green', linewidth=2, label='Expert (RL) Completion Rate')
        plt.fill_between(episodes_np, 
                              RL_done_rates_np - final_RL_done_std, 
                              RL_done_rates_np + final_RL_done_std, 
                              color='green', alpha=0.3)
    
    if len(IL_done_rates) > 0:
        plt.plot(episodes_np, IL_done_rates_np, color='purple', linewidth=2, label='IL Completion Rate')
        plt.fill_between(episodes_np, 
                              IL_done_rates_np - final_IL_done_std, 
                              IL_done_rates_np + final_IL_done_std, 
                              color='purple', alpha=0.3)
    
    plt.title('Final Task Completion Rates Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Completion Rate (%)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add total runtime to title if provided
    if total_minutes is not None and total_seconds is not None:
        plt.suptitle(f"Final Results - Total Runtime: {total_minutes}m {total_seconds}s - Imitation Learning - LR: {lr}, Layers: {num_layers}, Hidden: {hidden_dim}")
    else:
        plt.suptitle(f"Imitation Learning - LR: {lr}, Layers: {num_layers}, Hidden: {hidden_dim}")
    
    ts = time.time()
    timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    os.makedirs("/home/devtex/Documents/Genesis/graphs", exist_ok=True)
    plt.savefig(f"/home/devtex/Documents/Genesis/graphs/IL_{timestamp}.png", dpi=300, bbox_inches='tight')
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
    
    run_IL_test_suite(env, IL_agent, args)

def run_IL_test_suite(env, IL_agent, args):
    """Run test suite for IL model - 50 episodes"""
    print("\n" + "-" * 40)
    
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
        completion_rate = (IL_done_array.sum().item() / env.num_envs) * 100.0
        episode_completions.append(completion_rate)
        
        if (episode + 1) % 10 == 0:
            print(f"   Episode {episode+1}: Reward={IL_total_reward:.2f}, Completion={completion_rate:.1f}%")
    
    return episode_rewards, episode_completions

def plot_test_results(IL_rewards, IL_dones, args):
    """Plot test results with variance bands"""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    episodes = range(1, len(IL_rewards)+1)
    episodes_np = np.array(episodes)
    IL_rewards_np = np.array(IL_rewards)
    IL_dones_np = np.array(IL_dones)
    
    # Calculate variance bands for test results
    window_size = min(5, len(episodes))
    
    def calculate_test_std_bands(data_list, window_size):
        std_bands = []
        for i in range(len(data_list)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(data_list), i + window_size//2 + 1)
            
            if end_idx - start_idx > 1:
                window_data = data_list[start_idx:end_idx]
                std_bands.append(np.std(window_data))
            else:
                std_bands.append(0)
        return np.array(std_bands)
    
    rewards_std = calculate_test_std_bands(IL_rewards, window_size)
    dones_std = calculate_test_std_bands(IL_dones, window_size)

    # Plot 1: Episode Rewards with variance
    axes[0].plot(episodes_np, IL_rewards_np, color='blue', linewidth=2, alpha=0.8)
    axes[0].fill_between(episodes_np, 
                        IL_rewards_np - rewards_std, 
                        IL_rewards_np + rewards_std, 
                        color='blue', alpha=0.3, label='±1 STD')
    axes[0].axhline(y=sum(IL_rewards)/len(IL_rewards), color='red', linestyle='--', 
                   label=f'Average: {sum(IL_rewards)/len(IL_rewards):.1f}')
    axes[0].set_title('IL Agent - Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Completion Rates with variance
    axes[1].plot(episodes_np, IL_dones_np, color='green', linewidth=2, alpha=0.8)
    axes[1].fill_between(episodes_np, 
                        IL_dones_np - dones_std, 
                        IL_dones_np + dones_std, 
                        color='green', alpha=0.3, label='±1 STD')
    axes[1].axhline(y=sum(IL_dones)/len(IL_dones), color='red', linestyle='--', 
                   label=f'Average: {sum(IL_dones)/len(IL_dones):.1f}%')
    axes[1].set_title('IL Agent - Episode Completion Rates')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Completion Rate (%)')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f"IL Test Results - Task: {args.task}")
    plt.tight_layout()
    
    # Save the plot
    ts = time.time()
    timestamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(f"graphs/{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: graphs/IL_test_{timestamp}.png")
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
    
    if args.evaluate:
        print("Evaluate flag detected - switching to test mode")
        test_IL_model(args, lr, num_layers, hidden_dim)
    else:
        print("Starting IL training mode")
        train_imitation_learning(args, lr, num_layers, hidden_dim)
