import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import wandb

from model import AttnNet, RNDModel
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
    reward, ireward, value, returns, ivalue, ireturns, policyLoss, qValueLoss, iValueLoss, forwardLoss, entropy, gradNorm, \
        clipFrac, travel_dist, success_rate, explored_rate = tensorboardData
    metrics = {'Losses/Value': value,
               'Losses/Intrinsic Value': ivalue,
               'Losses/Policy Loss': policyLoss,
               'Losses/Q Value Loss': qValueLoss,
               'Losses/Intrinsic Value Loss': iValueLoss,
               'Losses/Forward Loss': forwardLoss,
               'Losses/Entropy': entropy,
               'Losses/Return': returns,
               'Losses/Intrinsic Return': ireturns,
               'Losses/Grad Norm': gradNorm,
               'Losses/Clip Frac': clipFrac,
               'Perf/Reward': reward,
               'Perf/Curiosity Reward': ireward,
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
    global_rnd_predictor = RNDModel(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_rnd_target = RNDModel(INPUT_DIM, EMBEDDING_DIM).to(device)

    # initialize optimizers
    combined_parameters = list(global_net.parameters()) + list(global_rnd_predictor.parameters())
    global_optimizer = optim.Adam(combined_parameters, lr=LR)

    # initialize decay (not use)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.96)

    curr_episode = 0

    if USE_WANDB:
        import parameter
        vars(parameter).__delitem__('__builtins__')
        wandb.init(project="Exploration", name=FOLDER_NAME, entity='ezo', config=vars(parameter), resume='allow',
                   id=WANDB_ID, notes=WANDB_NOTES)
        wandb.watch([global_net, global_rnd_predictor], log='all', log_freq=1000, log_graph=True)

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location=device)
        global_net.load_state_dict(checkpoint['model'])
        global_rnd_predictor.load_state_dict(checkpoint['predictor'])
        global_optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(global_optimizer.state_dict()['param_groups'][0]['lr'])

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    if device != local_device:
        network_weights = global_net.to(local_device).state_dict()
        rnd_predictor_weights = global_rnd_predictor.to(local_device).state_dict()
        rnd_target_weights = global_rnd_target.to(local_device).state_dict()
        global_net.to(device)
        global_rnd_predictor.to(device)
        global_rnd_target.to(device)
    else:
        network_weights = global_net.to(local_device).state_dict()
        rnd_predictor_weights = global_rnd_predictor.to(local_device).state_dict()
        rnd_target_weights = global_rnd_target.to(local_device).state_dict()

    # distributed training if multiple GPUs available
    dp_network = nn.DataParallel(global_net)
    dp_rnd_predictor = nn.DataParallel(global_rnd_predictor)
    dp_rnd_target = nn.DataParallel(global_rnd_target)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote([network_weights, rnd_predictor_weights, rnd_target_weights], curr_episode))
    
    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(17):
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
                    for i in range(17):
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
                logp_batch = torch.stack(rollouts[7]).to(device)
                ext_value_batch = torch.stack(rollouts[8]).to(device)
                reward_batch = torch.stack(rollouts[9]).to(device)
                ext_advantage_batch = torch.stack(rollouts[11]).to(device)
                ext_return_batch = torch.stack(rollouts[12]).to(device)
                int_value_batch = torch.stack(rollouts[13]).to(device)
                int_reward_batch = torch.stack(rollouts[14]).to(device)
                int_advantage_batch = torch.stack(rollouts[15]).to(device)
                int_return_batch = torch.stack(rollouts[16]).to(device)

                advantage_batch = 2.0 * ext_advantage_batch + 1.0 * int_advantage_batch

                # PPO
                for i in range(8):
                    predict_feature = dp_rnd_predictor(node_inputs_batch,
                                                       edge_inputs_batch,
                                                       current_inputs_batch,
                                                       node_padding_mask_batch,
                                                       edge_padding_mask_batch,
                                                       edge_mask_batch)
                    target_feature = dp_rnd_target(node_inputs_batch,
                                                   edge_inputs_batch,
                                                   current_inputs_batch,
                                                   node_padding_mask_batch,
                                                   edge_padding_mask_batch,
                                                   edge_mask_batch)
                    forward_loss = nn.MSELoss(reduction='none')(predict_feature, target_feature.detach()).mean(-1)
                    mask = torch.rand(forward_loss.shape).to(device)
                    mask = (mask < 0.25).type(torch.FloatTensor).to(device)
                    forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.tensor(1, device=device, dtype=torch.float32))
                    new_logp, new_ext_value, new_int_value = dp_network(node_inputs_batch,
                                                                        edge_inputs_batch,
                                                                        current_inputs_batch,
                                                                        node_padding_mask_batch,
                                                                        edge_padding_mask_batch,
                                                                        edge_mask_batch)
                    logp = torch.gather(new_logp, 1, action_batch).unsqueeze(1)
                    ratio = torch.exp(logp - logp_batch.detach())
                    surr1 = advantage_batch.detach() * ratio
                    surr2 = advantage_batch.detach() * ratio.clamp(1 - 0.2, 1 + 0.2)

                    policy_loss = -torch.min(surr1, surr2).mean()

                    ext_value_loss = nn.MSELoss()(new_ext_value, ext_return_batch.detach()).mean()
                    int_value_loss = nn.MSELoss()(new_int_value, int_return_batch.detach()).mean()
                    value_loss = ext_value_loss + int_value_loss

                    entropy = -(new_logp * new_logp.exp()).sum(dim=-1).mean()

                    total_loss = policy_loss + 0.2 * value_loss - 0.0 * entropy + forward_loss

                    global_optimizer.zero_grad()
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_net.parameters(), max_norm=1000, norm_type=2)
                    global_optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1).abs() > 0.2).float().mean()

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), int_reward_batch.mean().item(), ext_value_batch.mean().item(), ext_return_batch.mean().item(), int_value_batch.mean().item(), int_return_batch.mean().item(),
                        policy_loss.item(), ext_value_loss.item(), int_value_loss.item(), forward_loss.item(), -entropy.item(), grad_norm.item(), clip_frac.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
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
            if curr_episode % 64 == 0:
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
