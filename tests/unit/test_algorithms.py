import pytest
torch = pytest.importorskip("torch")
from bluerov_nav.networks import Actor, Value, TwinQ
from bluerov_nav.algorithms import advantages, policy_gradient, fit_value, OffPolicy, trpo_update

torch.set_num_threads(1)


def observations(n=2):
    return {"depth": torch.rand(n, 5, 128, 160),
            "goal": torch.rand(n, 5, 3), "range": torch.rand(n, 5, 1),
            "actions": torch.zeros(n, 5, 2)}


def test_time_limit_bootstrap_and_trace():
    r = torch.tensor([[1.], [2.]])
    v = torch.zeros_like(r)
    nv = torch.full_like(r, 10)
    _, returns = advantages(r, v, nv, torch.tensor([[0.], [1.]]),
                            torch.ones_like(r), .9, .95)
    torch.testing.assert_close(returns, torch.tensor([[10.], [2.]]))


@pytest.mark.parametrize("algorithm", ["a2c", "ppo", "trpo", "sac", "td3"])
def test_one_update_is_finite(algorithm):
    torch.manual_seed(42)
    obs = observations()
    actor = Actor()
    action, logp, latent = actor.sample(obs)
    assert action.shape == (2, 2) and torch.all(abs(action) <= 1)
    torch.testing.assert_close(logp, actor.log_prob(obs, latent))
    value = Value()
    adv = torch.tensor([[1.], [-1.]])
    if algorithm in {"sac", "td3"}:
        settings = dict(learning_rate=3e-5, gamma=.99, entropy_temperature=.2,
                        target_tau=.005, policy_delay=2, target_noise=.2, noise_clip=.5)
        learner = OffPolicy(algorithm, actor, TwinQ(), settings)
        batch = (obs, action.detach(), torch.ones(2, 1), observations(), torch.zeros(2, 1))
        result = learner.update(batch)
        if algorithm == "td3":
            assert "policy_loss" not in result
            result = learner.update(batch)
        assert "policy_loss" in result
        assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    elif algorithm == "trpo":
        result = trpo_update(actor, obs, latent.detach(), adv, .01, .1, cg_steps=2)
        assert result["kl"] <= .01
    else:
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-5)
        loss = policy_gradient(actor, optimizer, obs, latent.detach(), adv, logp.detach(),
                               .2 if algorithm == "ppo" else None)
        assert torch.isfinite(torch.tensor(loss))
        loss = fit_value(value, torch.optim.Adam(value.parameters(), lr=3e-5),
                         obs, torch.ones(2, 1), 0)
        assert torch.isfinite(torch.tensor(loss))
