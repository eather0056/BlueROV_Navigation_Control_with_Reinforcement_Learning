import argparse  # Import the argparse library for parsing command-line arguments.
import os  
import sys  # Import the sys library for accessing some variables used or maintained by the Python interpreter and functions that interact strongly with the interpreter.
import pickle  # Import the pickle library for serializing and de-serializing Python object structures, also called marshalling or flattening.
import time  
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import utilities and classes from custom modules.
from utils import *  # Import all from the utils module, which include functions and classes for data preprocessing, normalization, etc.
from models.mlp_policy import Policy  # Import the Policy class from the mlp_policy module, which defines the policy network architecture.
from models.mlp_critic import Value  # Import the Value class from the mlp_critic module, which defines the value network architecture.
from core.ppo import ppo_step  # Import the ppo_step function from the ppo module, which implements a single step of the PPO algorithm.
from core.common import estimate_advantages  # Import the estimate_advantages function from the common module, which computes advantage estimates for use in policy optimization.
from core.agent import Agent  # Import the Agent class from the agent module, which encapsulates the interaction between a policy and an environment.
from core.unity_underwater_env import Underwater_navigation  # Import the Underwater_navigation class from the unity_underwater_env module, which defines the underwater navigation environment.

# Create an argument parser object with a description of the script. take input from command line
parser = argparse.ArgumentParser(description='PyTorch PPO example')

# Argument for specifying the environment name to run. Default is "Hopper-v2".
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')

# Argument for specifying the path of a pre-trained model.
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')

# Argument for specifying whether to render the environment. Default is False.
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')

# Argument for specifying the log standard deviation for the policy. Default is -0.0.
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')

# Argument for specifying the discount factor (gamma) for rewards. Default is 0.99.
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')

# Argument for specifying the generalized advantage estimation parameter (tau). Default is 0.95.
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')

# Argument for specifying the L2 regularization strength. Default is 1e-3.
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')

# Argument for specifying the learning rate. Default is 3e-5.
parser.add_argument('--learning-rate', type=float, default=3e-5, metavar='G',
                    help='learning rate (default: 3e-5)')

# Argument for specifying whether randomization is enabled. Default is 1 (True).
parser.add_argument('--randomization', type=int, default=1, metavar='G')

# Argument for specifying whether adaptation is enabled. Default is 1 (True).
parser.add_argument('--adaptation', type=int, default=1, metavar='G')

# Argument for specifying the depth prediction model. Default is "dpt".
parser.add_argument('--depth-prediction-model', default="dpt", metavar='G')

# Argument for specifying the clipping epsilon for PPO. Default is 0.2.
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')

# Argument for specifying the number of consecutive history infos. Default is 4.
parser.add_argument('--hist-length', type=int, default=4, metavar='N',
                    help="the number of consecutive history infos (default: 4)")

# Argument for specifying the number of threads for the agent. Default is 1.
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')

# Argument for specifying the random seed. Default is 1.
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')

# Argument for specifying the minimal batch size per PPO update. Default is 2048.
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')

# Argument for specifying the minimal batch size for evaluation. Default is 2048.
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')

# Argument for specifying the maximal number of main iterations. Default is 200.
parser.add_argument('--max-iter-num', type=int, default=200, metavar='N',
                    help='maximal number of main iterations (default: 500)')

# Argument for specifying the interval between training status logs. Default is 1.
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')

# Argument for specifying the interval between saving model. Default is 0 (don't save).
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")

# Argument for specifying the GPU index to use. Default is 0.
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')

# Parse the command-line arguments and store them in 'args' object.
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = [] # Initialize an empty list to store environment instances.

# Create multiple instances of the Underwater_navigation environment based on the number of threads specified.
# Each environment instance is created with specific parameters
for i in range(args.num_threads):
    env.append(Underwater_navigation(args.depth_prediction_model, args.adaptation, args.randomization, i, args.hist_length))

# Extract observation dimensions from the first environment instance for further use.
img_depth_dim = env[0].observation_space_img_depth
goal_dim = env[0].observation_space_goal
ray_dim = env[0].observation_space_ray

# Check if the action space is discrete or continuous.
is_disc_action = len(env[0].action_space.shape) == 0

# Initialize a ZFilter object to normalize observations with specified clip value for depth values.
running_state = ZFilter(img_depth_dim, goal_dim, ray_dim, clip=30)

"""seeding"""
# Set random seeds for NumPy and PyTorch to ensure reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# env.seed(args.seed)

"""define actor and critic"""
# Define actor (policy) and critic (value) networks.
if args.model_path is None:
    # If no pre-trained model is provided, initialize policy and value networks from scratch.
    policy_net = Policy(args.hist_length, env[0].action_space.shape[0], log_std=args.log_std)
    value_net = Value(args.hist_length)
else:
    # If a pre-trained model path is provided, load the policy, value networks, and running state from the saved model file.
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))

# Move policy and value networks to the specified device (GPU or CPU).
policy_net.to(device)
value_net.to(device)

# Initialize Adam optimizers for policy and value networks with the specified learning rate.
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

# Define the number of optimization epochs and batch size for PPO.
optim_epochs = 10
optim_batch_size = 64

"""create agent"""
# Create an agent object using the Agent class, which encapsulates the interaction between the policy and the environment.
# - policy_net: The policy network used by the agent to select actions.
# - running_state: Object used for normalization of observations.
# - num_threads: Number of threads used for parallel environment interaction.
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads)

def update_params(batch, i_iter):
    # Convert numpy arrays from the batch to PyTorch tensors and move them to the appropriate device (GPU or CPU).
    imgs_depth = torch.from_numpy(np.stack(batch.img_depth)).to(dtype).to(device)
    goals = torch.from_numpy(np.stack(batch.goal)).to(dtype).to(device)
    rays = torch.from_numpy(np.stack(batch.ray)).to(dtype).to(device)
    hist_actions = torch.from_numpy(np.stack(batch.hist_action)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    
    # Calculate the values and fixed log probabilities using the value and policy networks.
    with torch.no_grad():
        values = value_net(imgs_depth, goals, rays, hist_actions)
        fixed_log_probs = policy_net.get_log_prob(imgs_depth, goals, rays, hist_actions, actions)

    # Estimate advantages and returns using the rewards, masks, and values.
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    # Perform mini-batch PPO update for multiple epochs.
    optim_iter_num = int(math.ceil(imgs_depth.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        # Shuffle indices for mini-batch updates.
        perm = np.arange(imgs_depth.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        # Apply permutation to the tensors.
        imgs_depth, goals, rays, hist_actions, actions, returns, advantages, fixed_log_probs = \
            imgs_depth[perm].clone(), goals[perm].clone(), rays[perm].clone(), hist_actions[perm].clone(), actions[perm].clone(),\
            returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        # Perform mini-batch updates.
        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, imgs_depth.shape[0]))
            imgs_depth_b, goals_b, rays_b, hist_actions_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                imgs_depth[ind], goals[ind], rays[ind], hist_actions[ind], \
                actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            # Call the PPO step function to update the policy and value networks.
            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, imgs_depth_b,
                     goals_b, rays_b, hist_actions_b, actions_b, returns_b, advantages_b,
                     fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


def main_loop():
    # Iterate over a specified number of iterations.
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        # Collect samples (trajectories) from the environment using the agent.
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
        t0 = time.time()  # Record the start time for update_params.

        # Update the parameters of the policy and value networks using the collected batch of samples.
        update_params(batch, i_iter)
        t1 = time.time()  # Record the end time for update_params.

        """evaluate with determinstic action (remove noise for exploration)"""
        # If specified, collect evaluation samples using the agent with deterministic actions (no exploration noise).
        if args.eval_batch_size > 0:
            _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        t2 = time.time()  # Record the end time for evaluation.

        # Print training status logs at specified intervals.
        if i_iter % args.log_interval == 0:
            if args.eval_batch_size > 0:
                print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                    i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))
            else:
                print(
                '{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\t'.format(
                    i_iter, log['sample_time'], t1 - t0, t2 - t1, log['min_reward'], log['max_reward'], log['avg_reward']))

        # Write training statistics to a text file.
        if args.randomization == 1:
            if args.adaptation == 1:
                my_open = open(os.path.join(assets_dir(), 'learned_models/{}_ppo_adapt.txt'.format(args.env_name)), "a")
            else:
                my_open = open(os.path.join(assets_dir(), 'learned_models/{}_ppo_rand.txt'.format(args.env_name)), "a")
        else:
            my_open = open(os.path.join(assets_dir(), 'learned_models/{}_ppo_norand.txt'.format(args.env_name)), "a")
        data = [str(i_iter), " ", str(log['avg_reward']), " ", str(log['num_episodes']),
                " ", str(log['ratio_success']), " ", str(log['avg_steps_success']), " ", str(log['avg_last_reward']), "\n"]
        for element in data:
            my_open.write(element)
        my_open.close()

        # Save the policy and value networks at specified intervals.
        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            # Move networks to CPU before saving.
            to_device(torch.device('cpu'), policy_net, value_net)
            if args.randomization == 1:
                if args.adaptation == 1:
                    # Save the networks with adaptation suffix in the file name.
                    pickle.dump((policy_net, value_net, running_state),
                                open(os.path.join(assets_dir(), 'learned_models/{}_ppo_adapt.p'.format(args.env_name)),
                                     'wb'))
                else:
                    # Save the networks with randomization suffix in the file name.
                    pickle.dump((policy_net, value_net, running_state),
                                open(os.path.join(assets_dir(), 'learned_models/{}_ppo_rand.p'.format(args.env_name)),
                                     'wb'))
            else:
                # Save the networks with no randomization suffix in the file name.
                pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo_norand.p'.format(args.env_name)), 'wb'))
            # Move networks back to the specified device after saving.
            to_device(device, policy_net, value_net)

        # Clean up GPU memory to avoid memory leaks.
        torch.cuda.empty_cache()

if __name__ == '__main__':
    # Set the start method for multiprocessing to 'spawn'.
    torch.multiprocessing.set_start_method('spawn')
    # Call the main loop function to start training.
    main_loop()
    print("Training finished.")

