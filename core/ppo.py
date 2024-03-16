import torch
# device = torch.device('cuda')
def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, imgs_depth,
             goals, rays, hist_actions, actions, returns, advantages, fixed_log_probs, clip_epsilon, l2_reg, device = torch.device('cuda')):

    """update critic"""
    total_value_loss = 0 # Initialize to accumulate value loss if optim_value_iternum > 1
    for _ in range(optim_value_iternum):
        values_pred = value_net(imgs_depth, goals, rays, hist_actions).to(device)
        value_loss = (values_pred - returns).pow(2).mean().to(device)
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()
        total_value_loss += value_loss.item() # Accumulate value loss

    """update policy"""
    log_probs = policy_net.get_log_prob(imgs_depth, goals, rays, hist_actions, actions)
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

    # optim_value_iternum is always >=1, otherwise, it divide by zero
    avg_value_loss = total_value_loss / optim_value_iternum

    return policy_surr.item(), avg_value_loss
