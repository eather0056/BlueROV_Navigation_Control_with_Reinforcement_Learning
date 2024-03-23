import argparse
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import utilities and classes from custom modules.
from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from core.trpo import trpo_step
from models.mlp_policy_disc import DiscretePolicy
from core.common import estimate_advantages
from core.agent import Agent
from core.unity_underwater_env import Underwater_navigation  # Import the Underwater_navigation class from the unity_underwater_env module, which defines the underwater navigation environment.
import wandb

# Create an argument parser object with a description of the script. take input from command line
parser = argparse.ArgumentParser(description='PyTorch TRPO example')

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

# Argument for specifying whether randomization is enabled. Default is 1 (True).
parser.add_argument('--randomization', type=int, default=1, metavar='G')

# Argument for specifying whether adaptation is enabled. Default is 1 (True).
parser.add_argument('--adaptation', type=int, default=1, metavar='G')

# Argument for specifying the number of consecutive history infos. Default is 4.
parser.add_argument('--hist-length', type=int, default=4, metavar='N',
                    help="the number of consecutive history infos (default: 4)")

# Argument for specifying the depth prediction model. Default is "dpt".
parser.add_argument('--depth-prediction-model', default="dpt", metavar='G')

parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')

parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')

# Argument for specifying the number of threads for the agent. Default is 4.
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')

# Argument for specifying the random seed. Default is 1.
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')

# Argument for specifying the minimal batch size per TRPO update. Default is 2048.
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')

# Argument for specifying the minimal batch size for evaluation. Default is 2048.
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')

# Argument for specifying the maximal number of main iterations. Default is 200.
parser.add_argument('--max-iter-num', type=int, default=200, metavar='N',
                    help='maximal number of main iterations (default: 200)')

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

# Initialize wandb
wandb.init(project='bluerov_navigaion_conrol', entity='eather0056', config=args)

dtype = torch.float64
# torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = [] # Initialize an empty list to store environment instances.

# state_dim = env.observation_space.shape[0]
# is_disc_action = len(env.action_space.shape) == 0
# running_state = ZFilter((state_dim,), clip=5)
# # running_reward = ZFilter((1,), demean=False, clip=10)

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
policy_net.to(dtype).to(device)
value_net.to(dtype).to(device)

"""create agent"""
# Create an agent object using the Agent class, which encapsulates the interaction between the policy and the environment.
# - policy_net: The policy network used by the agent to select actions.
# - running_state: Object used for normalization of observations.
# - num_threads: Number of threads used for parallel environment interaction.
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads)

def update_params(batch):
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

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform TRPO update"""
    success, total_value_loss, loss = trpo_step(policy_net.to(dtype).to(device), value_net.to(dtype).to(device), imgs_depth.to(dtype).to(device),
                     goals.to(dtype).to(device), rays.to(dtype).to(device), hist_actions.to(dtype).to(device), actions.to(dtype).to(device), returns.to(dtype).to(device), advantages.to(dtype).to(device), args.max_kl, args.damping, args.l2_reg, device)
    
    avg_policy_loss = loss
    avg_value_loss = total_value_loss

    return avg_policy_loss, avg_value_loss

def main_loop():
    avgrage_rewards = []
    policy_losses = []
    value_losses = []

    # Iterate over a specified number of iterations.
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
        t0 = time.time()

        # Update the parameters of the policy and value networks using the collected batch of samples.
        policy_loss, value_loss = update_params(batch)
        t1 = time.time()  # Record the end time for update_params.

        """evaluate with determinstic action (remove noise for exploration)"""
        # If specified, collect evaluation samples using the agent with deterministic actions (no exploration noise).
        if args.eval_batch_size > 0:
            _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        t2 = time.time()  # Record the end time for evaluation.

        # Print training status logs at specified intervals.
        if i_iter % args.log_interval == 0:
            log_data = {
                'Iteration_Number': i_iter,
                'Sample_Time_Sec': log['sample_time'],
                'Update_Duration_Sec': t1 - t0,
                'Evaluation_Duration_Sec': t2 - t1,
                'Policy Loss': policy_loss,
                'Value Loss': value_loss,
                'Training_Reward_Minimum': log['min_reward'],
                'Training_Reward_Maximum': log['max_reward'],
                'Training_Reward_Average': log['avg_reward'],
                'Number_of_Episodes': log['num_episodes'],
                'Success_Ratio': log.get('ratio_success', 0),
                'Average_Steps_Per_Success': log.get('avg_steps_success', 0),
                'Average_Last_Reward': log.get('avg_last_reward', 0)
            }

            if args.eval_batch_size > 0:
                log_data['eval_R_avg'] = log_eval['avg_reward']
                print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                    i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))
            else:
                print(
                '{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\t'.format(
                    i_iter, log['sample_time'], t1 - t0, t2 - t1, log['min_reward'], log['max_reward'], log['avg_reward']))
                
            wandb.log(log_data) # Log metrics to wandb 

        # Write training statistics to a text file.
        if args.randomization == 1:
            if args.adaptation == 1:
                my_open = open(os.path.join(assets_dir(), 'learned_models/{}_trpo_adapt_C.txt'.format(args.env_name)), "a")
            else:
                my_open = open(os.path.join(assets_dir(), 'learned_models/{}_trpo_rand_C.txt'.format(args.env_name)), "a")
        else:
            my_open = open(os.path.join(assets_dir(), 'learned_models/{}_trpo_norand_C.txt'.format(args.env_name)), "a")
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
                                open(os.path.join(assets_dir(), 'learned_models/{}_trpo_adapt_C.p'.format(args.env_name)),
                                     'wb'))
                else:
                    # Save the networks with randomization suffix in the file name.
                    pickle.dump((policy_net, value_net, running_state),
                                open(os.path.join(assets_dir(), 'learned_models/{}_trpo_rand_C.p'.format(args.env_name)),
                                     'wb'))
            else:
                # Save the networks with no randomization suffix in the file name.
                pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_trpo_norand_C.p'.format(args.env_name)), 'wb'))
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