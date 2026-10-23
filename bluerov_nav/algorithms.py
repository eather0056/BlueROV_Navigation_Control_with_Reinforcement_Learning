"""A2C, clipped PPO, KL-constrained TRPO, SAC, and TD3 update routines."""
import copy
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


def advantages(rewards, values, next_values, terminated, episode_ends, gamma, gae_lambda):
    """Bootstrap time limits but stop the GAE trace across every episode boundary."""
    with torch.no_grad():
        delta = rewards + gamma * (1 - terminated) * next_values - values
        result = torch.zeros_like(delta)
        carry = torch.zeros_like(delta[0])
        for i in reversed(range(len(delta))):
            carry = delta[i] + gamma * gae_lambda * (1 - episode_ends[i]) * carry
            result[i] = carry
        returns = result + values
        normalized = (result - result.mean()) / result.std(unbiased=False).clamp_min(1e-8)
        return normalized, returns


def fit_value(value, optimizer, obs, returns, l2_reg):
    loss = (value(obs) - returns.detach()).square().mean()
    loss = loss + l2_reg * sum(p.square().sum() for p in value.parameters())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def policy_gradient(actor, optimizer, obs, latent, adv, old_log_prob, clip=None):
    log_prob = actor.log_prob(obs, latent.detach())
    if clip is None:
        loss = -(log_prob * adv.detach()).mean()
    else:
        ratio = (log_prob - old_log_prob.detach()).exp()
        loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 40)
    optimizer.step()
    return float(loss.detach())


def _flat(tensors):
    return torch.cat([x.reshape(-1) for x in tensors])


def _set_params(params, vector):
    with torch.no_grad():
        offset = 0
        for p in params:
            p.copy_(vector[offset:offset + p.numel()].view_as(p))
            offset += p.numel()


def trpo_update(actor, obs, latent, adv, max_kl, damping, cg_steps=10, backtracks=10):
    """Backtracking accepts only an improved surrogate AND a satisfied KL bound."""
    params = list(actor.parameters())
    old_dist = actor.distribution(obs)
    old_dist = Normal(old_dist.loc.detach(), old_dist.scale.detach())
    old_log = actor.log_prob(obs, latent).detach()
    before = _flat([p.detach().clone() for p in params])

    def objective():
        return ((actor.log_prob(obs, latent) - old_log).exp() * adv).mean()

    def divergence():
        # Tanh is shared/invertible, so pre-transform KL is also action-space KL.
        return kl_divergence(old_dist, actor.distribution(obs)).sum(-1).mean()

    initial = objective()
    g = _flat(torch.autograd.grad(initial, params)).detach()
    if not torch.isfinite(g).all() or g.norm() < 1e-12:
        return {"policy_loss": -float(initial.detach()), "kl": 0.0, "accepted": False}

    def hvp(vector):
        grad_kl = _flat(torch.autograd.grad(divergence(), params, create_graph=True))
        h = _flat(torch.autograd.grad((grad_kl * vector).sum(), params)).detach()
        return h + damping * vector

    x, r, p = torch.zeros_like(g), g.clone(), g.clone()
    rr = r.dot(r)
    for _ in range(cg_steps):
        hp = hvp(p)
        denominator = p.dot(hp)
        if denominator <= 0 or not torch.isfinite(denominator):
            break
        alpha = rr / denominator
        x = x + alpha * p
        r = r - alpha * hp
        new_rr = r.dot(r)
        if new_rr < 1e-12:
            break
        p = r + (new_rr / rr) * p
        rr = new_rr
    curvature = x.dot(hvp(x))
    accepted, final_kl = False, 0.0
    if torch.isfinite(curvature) and curvature > 0:
        step = x * (2 * max_kl / curvature).sqrt()
        for j in range(backtracks):
            _set_params(params, before + (0.5 ** j) * step)
            with torch.no_grad():
                score, kl = objective(), divergence()
            if torch.isfinite(score) and torch.isfinite(kl) and score > initial and kl <= max_kl:
                accepted, final_kl = True, float(kl)
                break
    if not accepted:
        _set_params(params, before)
    with torch.no_grad():
        loss = -float(objective())
    return {"policy_loss": loss, "kl": final_kl, "accepted": accepted}


@torch.no_grad()
def soft_update(target, source, tau):
    for target_p, source_p in zip(target.parameters(), source.parameters()):
        target_p.lerp_(source_p, tau)


class OffPolicy:
    def __init__(self, algorithm, actor, critic, settings):
        self.algorithm, self.actor, self.critic = algorithm, actor, critic
        self.target_q = copy.deepcopy(critic).requires_grad_(False)
        self.target_actor = copy.deepcopy(actor).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=settings["learning_rate"])
        self.q_optimizer = torch.optim.Adam(critic.parameters(), lr=settings["learning_rate"])
        self.settings, self.updates = settings, 0

    def update(self, batch):
        obs, action, reward, nxt, terminated = batch
        s = self.settings
        with torch.no_grad():
            if self.algorithm == "sac":
                next_action, log_prob, _ = self.actor.sample(nxt)
                q1, q2 = self.target_q(nxt, next_action)
                target_q = torch.minimum(q1, q2) - s["entropy_temperature"] * log_prob
            else:
                noise = (torch.randn_like(action) * s["target_noise"]).clamp(
                    -s["noise_clip"], s["noise_clip"])
                next_action = (self.target_actor.deterministic(nxt) + noise).clamp(-1, 1)
                q1, q2 = self.target_q(nxt, next_action)
                target_q = torch.minimum(q1, q2)
            target = reward + s["gamma"] * (1 - terminated) * target_q
        q1, q2 = self.critic(obs, action)
        q_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self.updates += 1
        output = {"q_loss": float(q_loss.detach())}
        if self.algorithm == "sac" or self.updates % s["policy_delay"] == 0:
            self.critic.requires_grad_(False)
            try:
                if self.algorithm == "sac":
                    chosen, log_prob, _ = self.actor.sample(obs)
                    q1, q2 = self.critic(obs, chosen)
                    loss = (s["entropy_temperature"] * log_prob - torch.minimum(q1, q2)).mean()
                else:
                    chosen = self.actor.deterministic(obs)
                    loss = -self.critic.q1(obs, chosen).mean()
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
            finally:
                self.critic.requires_grad_(True)
            soft_update(self.target_q, self.critic, s["target_tau"])
            soft_update(self.target_actor, self.actor, s["target_tau"])
            output["policy_loss"] = float(loss.detach())
        return output
