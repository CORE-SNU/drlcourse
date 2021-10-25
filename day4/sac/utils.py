import os
import numpy as np

def get_env_spec(env):
    print('environment : ' + env.unwrapped.spec.id)
    print('obs dim : ', env.observation_space.shape, '/ ctrl dim : ', env.action_space.shape)
    dimS = env.observation_space.shape[0]
    dimA = env.action_space.shape[0]
    ctrl_range = env.action_space.high[0]
    max_ep_len = env._max_episode_steps
    print('-' * 80)

    print('ctrl range : ({:.2f}, {:.2f})'.format(-ctrl_range, ctrl_range))
    print('max_ep_len : ', max_ep_len)
    print('-' * 80)

    return dimS, dimA, ctrl_range, max_ep_len


def set_log_dir(env_id):
    if not os.path.exists('./train_log/'):
        os.mkdir('./train_log/')
    if not os.path.exists('./eval_log/'):
        os.mkdir('./eval_log/')

    if not os.path.exists('./train_log/' + env_id + '/'):
        os.mkdir('./train_log/' + env_id + '/')
    if not os.path.exists('./eval_log/' + env_id + '/'):
        os.mkdir('./eval_log/' + env_id + '/')

    if not os.path.exists('./checkpoints/'):
        os.mkdir('./checkpoints')

    if not os.path.exists('./checkpoints/' + env_id + '/'):
        os.mkdir('./checkpoints/' + env_id + '/')
    return


def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)
