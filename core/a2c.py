import torch

def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, imgs_depth,
             goals, rays, hist_actions, actions, returns, advantages, l2_reg, device = torch.device('cuda')):

    """update critic"""
    total_value_loss = 0 # Initialize to accumulate value loss if optim_value_iternum > 1
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
    policy_loss = -(log_probs * advantages).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

    return policy_loss.item(), total_value_loss

