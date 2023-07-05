import copy
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from env import Env
from parameter import *


class Worker:
    def __init__(self, meta_agent_id, policy_net, q_net, global_step, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, k_size=self.k_size, plot=save_image)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        self.current_node_index = 0
        self.travel_dist = 0
        self.robot_position = self.env.start_position

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(19):
            self.episode_buffer.append([])

    def get_observations(self):
        # get observations
        node_coords = copy.deepcopy(self.env.node_coords)
        graph = copy.deepcopy(self.env.graph)
        node_utility = copy.deepcopy(self.env.node_utility)
        guidepost = copy.deepcopy(self.env.guidepost)
        node_frontier_distribution = copy.deepcopy(self.env.node_frontier_distribution)

        # normalize observations
        node_coords = node_coords / 640
        node_utility = node_utility / 50

        # transfer to node inputs tensor
        n_nodes = node_coords.shape[0]
        node_utility_inputs = node_utility.reshape((n_nodes, 1))
        node_inputs = np.concatenate((node_coords, node_utility_inputs, guidepost), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)

        # padding the number of node to a given node padding size
        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)

        node_frontier_distribution = torch.FloatTensor(node_frontier_distribution / 1).unsqueeze(0).to(self.device)
        node_frontier_distribution = padding(node_frontier_distribution)

        # calculate a mask to padded nodes
        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # get the node index of the current robot position
        current_node_index = self.env.find_index_from_coords(self.robot_position)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        # padding edge mask
        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)

        edge = edge_inputs[current_index]
        while len(edge) < self.k_size:
            edge.append(0)

        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

        # calculate a mask for the padded edges (denoted by 0)
        edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)
        # neighbor_frontier_distribution = torch.gather(node_frontier_distribution, 1, edge_inputs.repeat(1, 36, 1)).permute(0, 2, 1)
        neighbor_gaze_candid = self.select_gaze_candidate(edge)
        observations = node_inputs, edge_inputs, current_index, node_frontier_distribution, neighbor_gaze_candid, \
            node_padding_mask, edge_padding_mask, edge_mask
        return observations

    def select_gaze_candidate(self, edge):
        neighbor_gaze_candidate = []
        half_window_size = 4
        for node in edge:
            vector = self.env.node_frontier_distribution[node]
            window = np.concatenate((vector[-half_window_size:], vector, vector[:half_window_size]))
            indices = np.arange(len(vector)) + half_window_size
            sum_vector = np.sum(np.take(window, indices.reshape(-1, 1) + np.arange(-half_window_size, half_window_size + 1)), axis=1)
            top_n_indices = np.argsort(-sum_vector)[:GAZE_SIZE]
            onehots = torch.zeros(GAZE_SIZE, 36)
            for i in range(-half_window_size, half_window_size+1):
                indices = (top_n_indices + i) % 36
                onehots += F.one_hot(torch.tensor(indices), num_classes=36).float()
            neighbor_gaze_candidate.append(onehots)
        neighbor_gaze_candidate = torch.stack(neighbor_gaze_candidate).unsqueeze(0).to(self.device)
        return neighbor_gaze_candidate  # 1, K, GAZE_SIZE, 36

    def select_node(self, observations):
        node_inputs, edge_inputs, current_index, node_frontier_distribution, neighbor_gaze_candid, node_padding_mask, \
            edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_frontier_distribution,
                                              neighbor_gaze_candid, node_padding_mask, edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        motion_index = action_index.item() // GAZE_SIZE
        next_node_index = edge_inputs[0, 0, motion_index]
        next_position = self.env.node_coords[next_node_index]

        next_gaze_index = action_index.item() % GAZE_SIZE

        return next_position, next_gaze_index, action_index

    def save_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_frontier_distribution, neighbor_gaze_candid, node_padding_mask, \
            edge_padding_mask, edge_mask = observations
        self.episode_buffer[0] += copy.deepcopy(node_inputs)
        self.episode_buffer[1] += copy.deepcopy(edge_inputs)
        self.episode_buffer[2] += copy.deepcopy(current_index)
        self.episode_buffer[3] += copy.deepcopy(node_frontier_distribution)
        self.episode_buffer[4] += copy.deepcopy(neighbor_gaze_candid)
        self.episode_buffer[5] += copy.deepcopy(node_padding_mask)
        self.episode_buffer[6] += copy.deepcopy(edge_padding_mask)
        self.episode_buffer[7] += copy.deepcopy(edge_mask)

    def save_action(self, action_index):
        self.episode_buffer[8] += action_index.unsqueeze(0).unsqueeze(0)

    def save_reward_done(self, reward, done):
        self.episode_buffer[9] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[10] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, current_index, node_frontier_distribution, neighbor_gaze_candid, node_padding_mask, \
            edge_padding_mask, edge_mask = observations
        self.episode_buffer[11] += copy.deepcopy(node_inputs)
        self.episode_buffer[12] += copy.deepcopy(edge_inputs)
        self.episode_buffer[13] += copy.deepcopy(current_index)
        self.episode_buffer[14] += copy.deepcopy(node_frontier_distribution)
        self.episode_buffer[15] += copy.deepcopy(neighbor_gaze_candid)
        self.episode_buffer[16] += copy.deepcopy(node_padding_mask)
        self.episode_buffer[17] += copy.deepcopy(edge_padding_mask)
        self.episode_buffer[18] += copy.deepcopy(edge_mask)

    def run_episode(self, curr_episode):
        done = False

        observations = self.get_observations()
        for i in range(128):
            self.save_observations(observations)
            next_position, next_gaze_index, action_index = self.select_node(observations)

            self.save_action(action_index)
            reward, done, self.robot_position, self.travel_dist = self.env.step(self.robot_position, next_position,
                                                                                next_gaze_index, self.travel_dist)
            self.save_reward_done(reward, done)
 
            observations = self.get_observations()
            self.save_next_observations(observations)

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i, self.travel_dist)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = self.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def work(self, currEpisode):
        self.run_episode(currEpisode)

    def calculate_edge_mask(self, edge_inputs):
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
        return bias_matrix
    
    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_rate), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)


if __name__ == '__main__':
    from model import PolicyNet, QNet
    ep = 1
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    # checkpoint = torch.load(model_path + '/checkpoint.pth')
    # policy_net.load_state_dict(checkpoint['policy_model'])
    q_net = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    worker = Worker(0, policy_net, q_net, ep, device, save_image=False)
    worker.work(ep)

