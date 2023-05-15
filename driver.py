import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
import wandb

from model import AttnNet
from runner import RLRunner
from parameter import *
from copy import deepcopy

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
    reward, value, returns, policyLoss, qValueLoss, entropy, gradNorm, travel_dist, success_rate, explored_rate = tensorboardData
    metrics = {'Losses/Value': value,
               'Losses/Policy Loss': policyLoss,
               'Losses/Q Value Loss': qValueLoss,
               'Losses/Entropy': entropy,
               'Losses/Grad Norm': gradNorm,
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
    global_net = AttnNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    old_net = AttnNet(INPUT_DIM, EMBEDDING_DIM).to(device).eval()

    # initialize optimizers
    global_optimizer = optim.Adam(global_net.parameters(), lr=LR)

    # initialize decay (not use)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.96)

    curr_episode = 0

    if USE_WANDB:
        import parameter
        vars(parameter).__delitem__('__builtins__')
        wandb.init(project="Exploration", name=FOLDER_NAME, entity='ezo', config=vars(parameter), resume='allow',
                   id=WANDB_ID, notes=WANDB_NOTES)
        wandb.watch(global_net, log='all', log_freq=1000, log_graph=True)

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_net.load_state_dict(checkpoint['model'])
        global_optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(global_optimizer.state_dict()['param_groups'][0]['lr'])

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    old_net.load_state_dict(deepcopy(global_net.state_dict()))

    if device != local_device:
        network_weights = global_net.to(local_device).state_dict()
        global_net.to(device)
    else:
        network_weights = global_net.to(local_device).state_dict()

    # distributed training if multiple GPUs available
    dp_network = nn.DataParallel(global_net)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(network_weights, curr_episode))
    
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
                    _, logp_old, value, entropy = old_net(node_inputs_batch,
                                                          edge_inputs_batch,
                                                          current_inputs_batch,
                                                          node_padding_mask_batch,
                                                          edge_padding_mask_batch,
                                                          edge_mask_batch,
                                                          action=action_batch)
                    _, _, value_prime, _ = old_net(next_node_inputs_batch,
                                                   next_edge_inputs_batch,
                                                   next_current_inputs_batch,
                                                   next_node_padding_mask_batch,
                                                   next_edge_padding_mask_batch,
                                                   next_edge_mask_batch,
                                                   action=action_batch)
                    advantage = reward_batch + GAMMA * (1 - done_batch) * value_prime - value

                for i in range(8):
                    _, logp, predicted_value, _ = dp_network(node_inputs_batch,
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

                    mse_loss = nn.MSELoss()
                    value_loss = mse_loss(predicted_value.squeeze(1).mean(), target_v_batch.detach().mean())

                    # print(policy_loss.item(), value_loss.item())
                    total_loss = policy_loss + 0.1 * value_loss - entropy * 0.0

                    global_optimizer.zero_grad()
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_net.parameters(), max_norm=1000, norm_type=2)
                    # print(grad_norm.item())
                    global_optimizer.step()

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), predicted_value.mean().item(), target_v_batch.mean().item(), policy_loss.item(),
                        value_loss.mean().item(), entropy.item(), grad_norm.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            old_net.load_state_dict(deepcopy(global_net.state_dict()))
            if device != local_device:
                network_weights = global_net.to(local_device).state_dict()
                global_net.to(device)
            else:
                network_weights = global_net.to(local_device).state_dict()

            job_list = []
            for i, meta_agent in enumerate(meta_agents):
                curr_episode += 1
                job_list.append(meta_agent.job.remote(network_weights, curr_episode))

            # save the model
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"model": global_net.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict(),
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
