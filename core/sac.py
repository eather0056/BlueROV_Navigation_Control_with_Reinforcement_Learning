import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn




    


def sac_step_with_replay_buffer(Q_Net, policy_net, value_net, optimizer_policy, optimizer_value, temperature, imgs_depth_b, goals_b, rays_b, hist_actions_b, actions_b, returns_b, advantages_b, fixed_log_probs_b, clip_epsilon, l2_reg, device, args):
    
    policy_net.train()
    value_net.train()

    imgs_depth_b = torch.tensor(imgs_depth_b, device=device)
    goals_b = torch.tensor(goals_b, device=device)
    rays_b = torch.tensor(rays_b, device=device)
    hist_actions_b = torch.tensor(hist_actions_b, device=device)
    actions_b = torch.tensor(actions_b, device=device)
    returns_b = torch.tensor(returns_b, device=device)
    advantages_b = torch.tensor(advantages_b, device=device)
    fixed_log_probs_b = torch.tensor(fixed_log_probs_b, device=device)

    policy_surr_total = 0
    value_loss_total = 0

    num_samples = min(len(imgs_depth_b), args.capacity)

    for _ in range(args.gradient_steps):
        indices = np.random.choice(num_samples, args.min_batch_size, replace=True)
        bn_s = imgs_depth_b[indices]
        # bn_a = actions_b[indices].reshape(-1, 1)
        # bn_r = returns_b[indices].reshape(-1, 1)
        bn_a = actions_b[indices]
        bn_r = returns_b[indices]
        bn_s_ = goals_b[indices]
        # bn_d = rays_b[indices].reshape(-1, 1)
        bn_d = rays_b[indices]
        bn_h = hist_actions_b[indices]

        
        target_value = value_net(bn_s, bn_s_, bn_d, bn_h).to(device)
        # next_q_value = bn_r + (1 - bn_d) * args.gamma * target_value
        next_q_value = bn_r + args.gamma * target_value

        expected_value = value_net(imgs_depth_b, goals_b, rays_b, hist_actions_b).to(device)
        # expected_value = Q_Net(imgs_depth_b, goals_b, rays_b, hist_actions_b).to(device)
        # expected_Q = policy_net(bn_s, bn_s_, bn_d, bn_h)
        # expected_Q = Q_Net(bn_s, bn_a)
        expected_Q = Q_Net(bn_s, bn_s_, bn_d, bn_a)

        # sample_action, log_prob, _, _, _ = policy_net.get_log_prob(bn_s, bn_s_, bn_d, bn_h, bn_a)
        sample_action = policy_net.select_action(bn_s, bn_s_, bn_d, bn_h)
        # expected_new_Q = policy_net(bn_s, bn_s_, bn_d, sample_action)
        # expected_new_Q = policy_net(bn_s, bn_s_, bn_d, bn_h)
        # expected_new_Q = Q_Net(bn_s, sample_action)
        expected_new_Q = Q_Net(bn_s, bn_s_, bn_d, sample_action)

        log_prob = policy_net.get_log_prob(bn_s, bn_s_, bn_d, bn_h, bn_a)
        # print(expected_new_Q)
        next_value = expected_new_Q - log_prob

        # print("expected value", expected_value)
        # print("next value", expected_new_Q) 

        V_loss = F.mse_loss(expected_value, returns_b)
        Q_loss = F.mse_loss(expected_Q, next_q_value.detach())

        # log_policy_target = expected_new_Q.clone().detach() - expected_value
        # pi_loss = log_prob * (log_prob - log_policy_target.detach())
        pi_loss = log_prob

        optimizer_policy.zero_grad()
        pi_loss.mean().backward(retain_graph=True)
        nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        optimizer_policy.step()

        optimizer_value.zero_grad()
        V_loss.mean().backward(retain_graph=True)
        nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        optimizer_value.step()

        optimizer_policy.zero_grad()
        Q_loss.mean().backward(retain_graph=True)
        nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        optimizer_policy.step()

        policy_surr_total += pi_loss.mean().item()
        value_loss_total += V_loss.mean().item()

        # for target_param, param in zip(value_net.parameters(), policy_net.parameters()):
        #     target_param.data.copy_(target_param * (1 - args.tau) + param * args.tau)

    avg_policy_surr = policy_surr_total / args.gradient_steps
    avg_value_loss = value_loss_total / args.gradient_steps

    return avg_policy_surr, avg_value_loss
