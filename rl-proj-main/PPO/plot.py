import matplotlib.pyplot as plt
# import re # re was not used

def parse_log_file(filepath):
    """
    Parses the log file to extract training metrics, including actor and critic losses.
    Expects a 3-line format per episode:
    1. Episode X completed
    2. Mean Reward / Mean Deviation | Cumulative Reward
    3. Actor Loss | Critic Loss
    """
    episodes = []
    mean_rewards = []
    mean_devs = []
    cumulative_rewards = []
    actor_losses = []
    critic_losses = []

    current_episode_index = 0  # This will be the 0-indexed episode number for plotting

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file '{filepath}' not found.")
        return [], [], [], [], [], []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "completed" in line:
            # This line indicates an episode finished.
            # Expect reward line next, then loss line.
            if i + 2 < len(lines):  # Ensure there are enough lines for reward and loss
                reward_line_str = lines[i+1].strip()
                loss_line_str = lines[i+2].strip()

                try:
                    # Parse reward line
                    reward_part, dev_cum_part = reward_line_str.split(' / ')
                    dev_part, cum_reward_part = dev_cum_part.split(' | ')

                    mean_reward = float(reward_part.strip())
                    mean_dev = float(dev_part.strip())
                    # The cumulative reward in the log is 10_000_000.0 + np.sum(rewards)
                    # We might want to subtract the base if we want the actual sum of rewards for the episode
                    # For now, plotting as is from the log.
                    cumulative_reward = float(cum_reward_part.strip())

                    try:
                        # Parse loss line
                        a_loss_str, c_loss_str = loss_line_str.split(' | ')
                        a_loss = float(a_loss_str.strip())
                        c_loss = float(c_loss_str.strip())

                        # If both parsed successfully, append all data
                        mean_rewards.append(mean_reward)
                        mean_devs.append(mean_dev)
                        cumulative_rewards.append(cumulative_reward)
                        actor_losses.append(a_loss)
                        critic_losses.append(c_loss)
                        episodes.append(current_episode_index)
                        current_episode_index += 1
                        
                        i += 3  # Move past "completed", reward, and loss lines
                        continue

                    except ValueError as e_loss:
                        print(f"Warning: Could not parse loss line: '{loss_line_str}'. Error: {e_loss}. Skipping data for episode block starting with '{line}'.")
                        i += 3 # Skip this block (completed, reward, malformed loss)
                        continue
                    except Exception as e_loss_other:
                        print(f"An unexpected error occurred on loss line: '{loss_line_str}'. Error: {e_loss_other}. Skipping data for episode block.")
                        i += 3 # Skip this block
                        continue
                
                except ValueError as e_reward:
                    print(f"Warning: Could not parse reward line: '{reward_line_str}'. Error: {e_reward}. Skipping data for episode block starting with '{line}'.")
                    # Decide how to advance 'i'. If reward line is bad, the loss line is likely not for this 'completed' line.
                    # Incrementing by 1 will make the current reward_line_str be processed as a regular line in next iteration.
                    i += 1 
                    continue
                except Exception as e_reward_other:
                    print(f"An unexpected error occurred on reward line: '{reward_line_str}'. Error: {e_reward_other}. Skipping data for episode block.")
                    i += 1
                    continue
            else:
                # Not enough lines left for a full entry after a "completed" line
                # print(f"Warning: 'completed' line found near EOF without subsequent data lines: '{line}'")
                i += 1 # Move to the next line
                continue
        else:
            # Line is not "completed", just move to the next one
            i += 1
            
    return episodes, mean_rewards, mean_devs, cumulative_rewards, actor_losses, critic_losses

def plot_metrics(episodes, mean_rewards, mean_devs, cumulative_rewards, 
                 actor_losses, critic_losses, max_episodes_to_plot=None):
    """
    Generates and shows plots for the training metrics.
    max_episodes_to_plot: If specified, only plot up to this many episodes (0-indexed).
                          If None, plot all episodes.
    """
    if not episodes:
        print("No data to plot.")
        return

    if max_episodes_to_plot is not None and max_episodes_to_plot > 0:
        num_available_episodes = len(episodes)
        limit_index = min(max_episodes_to_plot, num_available_episodes) 
        
        if limit_index == 0:
            print("No episodes to plot with the given limit.")
            return

        print(f"Plotting data for the first {limit_index} episodes.")
        episodes = episodes[:limit_index]
        mean_rewards = mean_rewards[:limit_index]
        mean_devs = mean_devs[:limit_index]
        cumulative_rewards = cumulative_rewards[:limit_index]
        actor_losses = actor_losses[:limit_index]
        critic_losses = critic_losses[:limit_index]
    elif max_episodes_to_plot is not None and max_episodes_to_plot <=0:
        print(f"Invalid max_episodes_to_plot value: {max_episodes_to_plot}. Plotting all data or no data if 0.")
        if max_episodes_to_plot == 0:
            return

    if not episodes: 
        print("No data to plot after applying limit.")
        return

    fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True) # Increased to 5 subplots

    # Plot 1: Mean Reward with Mean Deviation as error bars
    axs[0].errorbar(episodes, mean_rewards, yerr=None, fmt='-o', capsize=5, ecolor='lightcoral', label='Mean Reward', errorevery=1) # Changed yerr=0 to yerr=mean_devs
    axs[0].set_ylabel("Reward Value")
    axs[0].set_title("Mean Reward per Episode (with Mean Deviation)")
    axs[0].grid(True)
    axs[0].legend()

    # Plot 2: Cumulative Reward
    axs[1].plot(episodes, cumulative_rewards, marker='o', linestyle='-', color='green', label='Cumulative Reward')
    axs[1].set_ylabel("Cumulative Reward")
    axs[1].set_title("Cumulative Reward over Episodes")
    axs[1].grid(True)
    axs[1].legend()
    
    # Plot 3: Mean Deviation
    axs[2].plot(episodes, mean_devs, marker='s', linestyle='-', color='purple', label='Mean Deviation')
    axs[2].set_ylabel("Mean Deviation")
    axs[2].set_title("Mean Deviation per Episode")
    axs[2].grid(True)
    axs[2].legend()

    # Plot 4: Actor Loss
    axs[3].plot(episodes, actor_losses, marker='.', linestyle='-', color='deepskyblue', label='Actor Loss')
    axs[3].set_ylabel("Actor Loss")
    axs[3].set_title("Actor Loss per Episode")
    axs[3].grid(True)
    axs[3].legend()

    # Plot 5: Critic Loss
    axs[4].plot(episodes, critic_losses, marker='.', linestyle='-', color='orangered', label='Critic Loss')
    axs[4].set_xlabel("Episode Number") # X-axis label on the last plot
    axs[4].set_ylabel("Critic Loss")
    axs[4].set_title("Critic Loss per Episode")
    axs[4].grid(True)
    axs[4].legend()

    fig.suptitle("Training Progress Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig("training_metrics_loss.png", dpi=300)
    print("Plot saved to training_metrics.png")
    # plt.show() # Uncomment if you want to display the plot as well

if __name__ == "__main__":
    log_filepath = "out2.txt" # Make sure this is your log file's name
    
    # Unpack the new lists for losses
    episodes, mean_rewards, mean_devs, cumulative_rewards, actor_losses, critic_losses = parse_log_file(log_filepath)
    
    if episodes:
        print(f"Parsed {len(episodes)} data points in total.")
        # You can uncomment these for debugging if needed:
        # print(f"First 5 Mean Rewards: {mean_rewards[:5]}")
        # print(f"First 5 Actor Losses: {actor_losses[:5]}")
        # print(f"First 5 Critic Losses: {critic_losses[:5]}")
        
        # Set this to None or remove the argument to plot all episodes
        # Set to a number to plot only the first N episodes
        num_episodes_to_show = None # Example: plot all parsed episodes
        # num_episodes_to_show = 200 # Example: plot first 200 episodes
        
        plot_metrics(episodes, mean_rewards, mean_devs, cumulative_rewards, 
                     actor_losses, critic_losses, # Pass the new loss lists
                     max_episodes_to_plot=num_episodes_to_show)
    else:
        print(f"No data could be parsed from '{log_filepath}'. Please check the file format and content.")