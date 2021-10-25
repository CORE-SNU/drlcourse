import time
import csv
import argparse
import gym
import torch

from sac_agent import SACAgent
from utils import get_env_spec, set_log_dir


def run_sac(
            env_id,
            max_iter=1e6,
            eval_interval=2000,
            start_train=10000,
            train_interval=50,
            buffer_size=1e6,
            fill_buffer=20000,
            truncate=1000,
            gamma=0.99,
            pi_lr=3e-4,
            q_lr=3e-4,
            polyak=5e-3,
            alpha=0.2,
            hidden1=256,
            hidden2=256,
            batch_size=128,
            device='cpu',
            render='False'
            ):

    params = locals()

    max_iter = int(max_iter)
    buffer_size = int(buffer_size)
    env = gym.make(env_id)

    dimS, dimA, ctrl_range, max_ep_len = get_env_spec(env)

    if truncate is not None:
        max_ep_len = truncate

    agent = SACAgent(
                     dimS,
                     dimA,
                     ctrl_range,
                     gamma=gamma,
                     pi_lr=pi_lr,
                     q_lr=q_lr,
                     polyak=polyak,
                     alpha=alpha,
                     hidden1=hidden1,
                     hidden2=hidden2,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     device=device,
                     render=render
                     )

    set_log_dir(env_id)

    num_checkpoints = 5
    checkpoint_interval = max_iter // (num_checkpoints - 1)
    current_time = time.strftime("%m%d-%H%M%S")
    train_log = open('./train_log/' + env_id + '/SAC_' + current_time + '.csv',
                     'w', encoding='utf-8', newline='')

    path = './eval_log/' + env_id + '/SAC_' + current_time
    eval_log = open(path + '.csv', 'w', encoding='utf-8', newline='')

    train_logger = csv.writer(train_log)
    eval_logger = csv.writer(eval_log)

    with open(path + '.txt', 'w') as f:
        for key, val in params.items():
            print(key, '=', val, file=f)

    obs = env.reset()
    step_count = 0
    ep_reward = 0

    # main loop
    start = time.time()
    for t in range(max_iter + 1):
        if t < fill_buffer:
            action = env.action_space.sample()
        else:
            action = agent.act(obs)

        next_obs, reward, done, _ = env.step(action)
        step_count += 1

        if step_count == max_ep_len:
            done = False

        agent.buffer.append(obs, action, next_obs, reward, done)

        obs = next_obs
        ep_reward += reward

        if done or (step_count == max_ep_len):
            train_logger.writerow([t, ep_reward])
            obs = env.reset()
            step_count = 0
            ep_reward = 0

        if (t >= start_train) and (t % train_interval == 0):
            for _ in range(train_interval):
                agent.train()

        if t % eval_interval == 0:
            eval_score = eval_agent(agent, env_id, render=False)
            log = [t, eval_score]
            print('step {} : {:.4f}'.format(t, eval_score))
            eval_logger.writerow(log)

        if t % (10 * eval_interval) == 0:
            if render:
                render_agent(agent, env_id)

        if t % checkpoint_interval == 0:
            agent.save_model('./checkpoints/' + env_id + '/sac_{}th_iter_'.format(t))

    train_log.close()
    eval_log.close()

    return


def render_agent(agent, env_id):
    eval_agent(agent, env_id, eval_num=1, render=True)


def eval_agent(agent, env_id, eval_num=5, render=False):
    log = []
    for ep in range(eval_num):
        env = gym.make(env_id)

        state = env.reset()
        step_count = 0
        ep_reward = 0
        done = False

        while not done:
            if render and ep == 0:
                env.render()

            action = agent.act(state, eval=True)
            next_state, reward, done, _ = env.step(action)
            step_count += 1
            state = next_state
            ep_reward += reward

        if render and ep == 0:
            env.close()
        log.append(ep_reward)

    avg = sum(log) / eval_num

    return avg



if __name__ == '__main__':

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--truncate', required=False, default=1000, type=int)
    parser.add_argument('--env', required=True)
    parser.add_argument('--device', required=False, default=default_device)
    parser.add_argument('--max_iter', required=False, default=5e5, type=float)
    parser.add_argument('--eval_interval', required=False, default=2000, type=int)
    parser.add_argument('--render', required=False, default=False, type=bool)
    parser.add_argument('--tau', required=False, default=5e-3, type=float)
    parser.add_argument('--lr', required=False, default=3e-4, type=float)
    parser.add_argument('--hidden1', required=False, default=256, type=int)
    parser.add_argument('--hidden2', required=False, default=256, type=int)
    parser.add_argument('--train_interval', required=False, default=50, type=int)
    parser.add_argument('--start_train', required=False, default=10000, type=int)
    parser.add_argument('--fill_buffer', required=False, default=20000, type=int)

    args = parser.parse_args()

    run_sac(
            args.env,
            max_iter=args.max_iter,
            eval_interval=args.eval_interval,
            start_train=args.start_train,
            train_interval=args.train_interval,
            fill_buffer=args.fill_buffer,
            truncate=args.truncate,
            gamma=0.99,
            pi_lr=args.lr,
            q_lr=args.lr,
            polyak=args.tau,
            alpha=0.2,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            batch_size=128,
            buffer_size=1e6,
            device=args.device,
            render=args.render
            )
