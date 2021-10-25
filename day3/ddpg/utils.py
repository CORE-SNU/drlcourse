import torch


def save_snapshot(path, actor, critic, target_actor, target_critic, actor_optim, critic_optim):
    # print('adding checkpoints...')
    checkpoint_path = path + 'model.pth.tar'
    torch.save(
        {'actor': actor.state_dict(),
         'critic': critic.state_dict(),
         'target_actor': target_actor.state_dict(),
         'target_critic': target_critic.state_dict(),
         'actor_optimizer': actor_optim.state_dict(),
         'critic_optimizer': critic_optim.state_dict()
         },
        checkpoint_path)
    return


def recover_snapshot(path, actor, critic, target_actor, target_critic, actor_optim, critic_optim, device):
    print('recovering snapshot...')
    checkpoint = torch.load(path, map_location=torch.device(device))

    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    target_actor.load_state_dict(checkpoint['target_actor'])
    target_critic.load_state_dict(checkpoint['target_critic'])
    actor_optim.load_state_dict(checkpoint['actor_optimizer'])
    critic_optim.load_state_dict(checkpoint['critic_optimizer'])
    return


def load_model(agent, path, device):
    print('loading pre-trained weight...')
    checkpoint = torch.load(path, map_location=torch.device(device))
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    return