import torch
import torch.nn.functional as F


def td3_update(
    actor,
    critic,
    actor_target,
    critic_target,
    replay_buffer,
    actor_optimizer,
    critic_optimizer,
    batch_size,
    gamma,
    tau,
    policy_noise,
    noise_clip,
    policy_delay,
    step
):
    batch = replay_buffer.sample(batch_size)

    obs = batch['obs']
    actions = batch['actions']
    rewards = batch['rewards']
    next_obs = batch['next_obs']
    dones = batch['dones']

    with torch.no_grad():
        noise = (
            torch.randn_like(actions) * policy_noise
        ).clamp(-noise_clip, noise_clip)

        next_actions = (
            actor_target(next_obs) + noise
        ).clamp(-1.0, 1.0)

        target_q1, target_q2 = critic_target(next_obs, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + gamma * (1.0 - dones) * target_q

    # -------- Critic update --------
    current_q1, current_q2 = critic(obs, actions)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = None

    # -------- Delayed Actor update --------
    if step % policy_delay == 0:
        actor_actions = actor(obs)
        actor_loss = -critic.q1(
            obs, actor_actions
        ).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # -------- Target networks --------
        for p, tp in zip(actor.parameters(), actor_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        for p, tp in zip(critic.parameters(), critic_target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    return {
        'critic_loss': critic_loss.item(),
        'actor_loss': actor_loss.item() if actor_loss is not None else None
    }
