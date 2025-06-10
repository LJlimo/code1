import torch
import numpy as np

'''经验回放池'''
class RolloutBuffer:
    def __init__(self, rollout_steps, num_cameras, num_targets, obs_shape, global_state_shape, device):
        '''
        rollout_steps: 回放池的大小
        num_cameras: 相机数量
        num_targets: 目标数量
        obs_shape: 观测的形状 (max_visual_num, obs_dim)
        global_state_shape: 每个agent全局状态的形状 (num_cameras, num_targets, state_dim) 这里要注意，传入的参数应该是这个形状的
        device: 设备类型
        '''
        self.rollout_steps = rollout_steps
        self.num_cameras = num_cameras
        self.num_targets = num_targets
        self.obs_shape = obs_shape
        self.global_state_shape = global_state_shape
        self.device = device
        self.clear()

    def insert(self, obs_tensor, obs_mask, global_state, actions, log_probs, rewards, values, dones):
        '''
        存储一条经验
        每个时间步插入N个相机的数据
        obs_tensor: 观测张量 [num_cameras, max_visual_num, obs_dim]
        global_state: 全局状态 [num_cameras, num_targets, state_dim]
        '''
        self.obs_tensor.append(obs_tensor)
        self.obs_mask.append(obs_mask)
        self.global_state.append(global_state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(values)
        self.dones.append(dones)

    def compute_returns_and_advantages(self, last_values, gamma=0.99, gae_lambda=0.95, normalize_adv=True):
        '''
        用rollout_steps步的数据，从后往前递归使用GAE计算advantage、return、last_value

        这个方法是实现计算advantage和returns的核心

        last_values: 上一个时间步的值
        gamma: 折扣因子
        gae_lambda: GAE的lambda参数
        normalize_adv: 是否对advantage进行归一化
        '''
        T = len(self.rewards)
        N = self.num_cameras 
        advantages = torch.zeros((T, N), device=self.device)
        returns = torch.zeros((T, N), device=self.device)
        last_adv = torch.zeros(N, device=self.device)

        for t in reversed(range(T)):
            next_value = last_values if t == T - 1 else self.values[t+1].detach()
            mask = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t].detach()
            last_adv = delta + gamma * gae_lambda * mask * last_adv
            advantages[t] = last_adv
            returns[t] = advantages[t] + self.values[t].detach()
        
        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages
        self.returns = returns

    def get_flattened_data(self):
        '''
        T: rollout_steps, 时间步数
        N: num_cameras, 相机数量
        L: max_visual_num, 观测长度
        M: num_targets, 目标数量
        D: obs_dim, 观测维度
        这个方法是将数据展平为[T*N, L, D]/[T*N, M, D]的形状，就是拼成一个大批量，便于后面并行训练
        '''
        obs_tensor = torch.stack(self.obs_tensor).reshape(-1, *self.obs_shape)  # [T*N, L, D]
        obs_mask = torch.stack(self.obs_mask).reshape(-1, self.obs_shape[0]) # [T*N, L]
        state = torch.stack(self.global_state).reshape(-1, self.global_state_shape[1], self.global_state_shape[2]) # [T*N, M, D]
        actions = torch.stack(self.actions).reshape(-1)
        log_probs = torch.stack(self.log_probs).reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)

        return obs_tensor, obs_mask, state, actions, log_probs, advantages, returns
        # 这里返回的是一个大批量的观测数据，形状为[T*N, ..., ...]
    
    def get_batches(self, batch_size, ppo_epochs):
        '''
        每个epoch随机打乱并返回
        batch_size: 批量大小
        ppo_epochs: PPO的epoch数
        '''
        obs_tensor, obs_mask, state, actions, log_probs, advantages, returns = self.get_flattened_data()
        total_samples = obs_tensor.size(0)
        
        for _ in range(ppo_epochs):
            indices = torch.randperm(total_samples) # 打乱所有样本顺序
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples) # 防止最后一个batch不满或越界
                idx = [indices[start:end]]
                yield obs_tensor[idx], obs_mask[idx], state[idx], actions[idx], log_probs[idx], advantages[idx], returns[idx]



    def clear(self):
        '''清空回放池'''
        self.obs_tensor = []
        self.obs_mask = []
        self.global_state = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = None
        self.returns = None
    
