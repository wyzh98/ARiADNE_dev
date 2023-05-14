import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
import wandb

from model import PolicyNet, CriticNet
from runner import RLRunner
from parameter import *

ray.init()
print("Welcome to RL autonomous exploration!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def writeToTensorBoard(writer, tensorboardData, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboardData = np.array(tensorboardData)
    tensorboardData = list(np.nanmean(tensorboardData, axis=0))
    reward, value, returns, policyLoss, qValueLoss, entropy, policyGradNorm, qValueGradNorm, travel_dist, success_rate, explored_rate = tensorboardData
    metrics = {'Losses/Value': value,
               'Losses/Policy Loss': policyLoss,
               'Losses/Q Value Loss': qValueLoss,
               'Losses/Entropy': entropy,
               'Losses/Policy Grad Norm': policyGradNorm,
               'Losses/Q Value Grad Norm': qValueGradNorm,
               'Perf/Reward': reward,
               'Perf/Travel Distance': travel_dist,
               'Perf/Explored Rate': explored_rate,
               'Perf/Success Rate': success_rate}
    for k, v in metrics.items():
        writer.add_scalar(k, v, curr_episode)
    if USE_WANDB:
        wandb.log(metrics, step=curr_episode)


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    # initialize neural networks
    global_policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    old_policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device).eval()
    global_critic_net = CriticNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_critic_optimizer = optim.Adam(global_critic_net.parameters(), lr=LR)

    # initialize decay (not use)
    policy_lr_decay = optim.lr_scheduler.StepLR(global_policy_optimizer, step_size=DECAY_STEP, gamma=0.96)
    critic_lr_decay = optim.lr_scheduler.StepLR(global_critic_optimizer, step_size=DECAY_STEP, gamma=0.96)

    curr_episode = 0

    if USE_WANDB:
        import parameter
        vars(parameter).__delitem__('__builtins__')
        wandb.init(project="Exploration", name=FOLDER_NAME, entity='ezo', config=vars(parameter), resume='allow',
                   id=WANDB_ID, notes=WANDB_NOTES)
        wandb.watch([global_policy_net, global_critic_net], log='all', log_freq=1000, log_graph=True)

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_critic_net.load_state_dict(checkpoint['critic_model'])
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        policy_lr_decay.load_state_dict(checkpoint['policy_lr_decay'])
        critic_lr_decay.load_state_dict(checkpoint['critic_lr_decay'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    old_policy_net.load_state_dict(global_policy_net.state_dict())
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        critic_weights = global_critic_net.to(local_device).state_dict()
        global_policy_net.to(device)
        global_critic_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
        critic_weights = global_critic_net.to(local_device).state_dict()
    weights_set.append(policy_weights)
    weights_set.append(critic_weights)

    # distributed training if multiple GPUs available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_critic = nn.DataParallel(global_critic_net)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))
    
    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(16):
        experience_buffer.append([])
    
    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list, num_returns=NUM_META_AGENT)
            # get the results
            done_jobs = ray.get(done_id)
            
            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            while len(experience_buffer[0]) >= BATCH_SIZE:

                if len(experience_buffer[0]) < BATCH_SIZE:
                    experience_buffer = []
                    for i in range(16):
                        experience_buffer.append([])
                    perf_metrics = {}
                    for n in metric_name:
                        perf_metrics[n] = []
                    break

                rollouts = []
                for i in range(len(experience_buffer)):
                    rollouts.append(experience_buffer[i][:BATCH_SIZE])
                for i in range(len(experience_buffer)):
                    experience_buffer[i] = experience_buffer[i][BATCH_SIZE:]

                # stack batch data to tensors
                node_inputs_batch = torch.stack(rollouts[0]).to(device)
                edge_inputs_batch = torch.stack(rollouts[1]).to(device)
                current_inputs_batch = torch.stack(rollouts[2]).to(device)
                node_padding_mask_batch = torch.stack(rollouts[3]).to(device)
                edge_padding_mask_batch = torch.stack(rollouts[4]).to(device)
                edge_mask_batch = torch.stack(rollouts[5]).to(device)
                action_batch = torch.stack(rollouts[6]).to(device)
                reward_batch = torch.stack(rollouts[7]).to(device)
                done_batch = torch.stack(rollouts[8]).to(device)
                next_node_inputs_batch = torch.stack(rollouts[9]).to(device)
                next_edge_inputs_batch = torch.stack(rollouts[10]).to(device)
                next_current_inputs_batch = torch.stack(rollouts[11]).to(device)
                next_node_padding_mask_batch = torch.stack(rollouts[12]).to(device)
                next_edge_padding_mask_batch = torch.stack(rollouts[13]).to(device)
                next_edge_mask_batch = torch.stack(rollouts[14]).to(device)
                target_v_batch = torch.stack(rollouts[15]).to(device)

                # PPO
                with torch.no_grad():
                    _, logp_old, entropy = old_policy_net(node_inputs_batch,
                                                          edge_inputs_batch,
                                                          current_inputs_batch,
                                                          node_padding_mask_batch,
                                                          edge_padding_mask_batch,
                                                          edge_mask_batch,
                                                          action=action_batch)
                    value_prime, _ = dp_critic(next_node_inputs_batch,
                                               next_current_inputs_batch,
                                               next_node_padding_mask_batch,
                                               next_edge_mask_batch)
                    value, _ = dp_critic(node_inputs_batch,
                                         current_inputs_batch,
                                         node_padding_mask_batch,
                                         edge_mask_batch)
                    advantage = reward_batch + GAMMA * (1 - done_batch) * value_prime - value

                for i in range(8):
                    _, logp, _ = dp_policy(node_inputs_batch,
                                           edge_inputs_batch,
                                           current_inputs_batch,
                                           node_padding_mask_batch,
                                           edge_padding_mask_batch,
                                           edge_mask_batch,
                                           action=action_batch)
                    ratio = torch.exp(logp - logp_old.detach())
                    surr1 = ratio * advantage.detach()
                    surr2 = ratio.clamp(1-0.2, 1+0.2) * advantage.detach()
                    policy_loss = -torch.min(surr1, surr2).mean()

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=1000)
                    global_policy_optimizer.step()

                    predicted_value, _ = dp_critic(node_inputs_batch,
                                                   current_inputs_batch,
                                                   node_padding_mask_batch,
                                                   edge_mask_batch)
                    mse_loss = nn.MSELoss()
                    value_loss = mse_loss(predicted_value.squeeze(1).mean(), target_v_batch.detach().mean())
                    global_critic_optimizer.zero_grad()
                    value_loss.backward()
                    value_grad_norm = torch.nn.utils.clip_grad_norm_(global_critic_net.parameters(), max_norm=2000)
                    global_critic_optimizer.step()

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), predicted_value.mean().item(), target_v_batch.mean().item(), policy_loss.item(),
                        value_loss.mean().item(), entropy.item(), policy_grad_norm.item(), value_grad_norm.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            old_policy_net.load_state_dict(global_policy_net.state_dict())
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                critic_weights = global_critic_net.to(local_device).state_dict()
                global_policy_net.to(device)
                global_critic_net.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
                critic_weights = global_critic_net.to(local_device).state_dict()
            weights_set.append(policy_weights)
            weights_set.append(critic_weights)

            job_list = []
            for i, meta_agent in enumerate(meta_agents):
                curr_episode += 1
                job_list.append(meta_agent.job.remote(weights_set, curr_episode))

            # save the model
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "critic_model": global_critic_net.state_dict(),
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "critic_optimizer": global_critic_optimizer.state_dict(),
                              "episode": curr_episode,
                              "policy_lr_decay": policy_lr_decay.state_dict(),
                              "critic_lr_decay": critic_lr_decay.state_dict(),
                              }
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
    
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
