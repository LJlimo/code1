import torch
import imageio
import os
import numpy as np


def make_gif(image_floder, output_file='output.gif', fps=10):
    image = []
    file_list = sorted(os.listdir(image_floder))
    for filename in file_list:
        if filename.endswith('.png'):
            img_path = os.path.join(image_floder, filename)
            image.append(imageio.imread(img_path))
    imageio.mimsave(output_file, image, fps=fps)

def norm_col_init(weights, std=1.0):
    '''对权重进行列归一化初始化'''
    x = torch.randn(weights.size())
    x *= std / torch.sqrt(((x ** 2).sum(1, keepdim=True)))
    return x 

def process_obs_list(obs_list, max_visual_num, obs_dim=2, device='cpu', strategy='nearest'):
    '''
    将原始变长的观测列表转化为固定大小的张量
    
    参数:
    obs_list: 原始观测列表
    max_visual_num: 目标的最大可视数量
    obs_dim: 观测维度[距离， 角度]
    staregy: 超过max_visual_num的目标裁剪策略 如'nearest'、'random'  这部分暂时这么写，之后可能用势能来做

    返回:
    obs_tensor: (num_cameras, max_visual_num, obs_dim)处理后的观测张量
    obs_mask: (num_cameras, max_visual_num) 处理后的观测掩码, 表明哪些目标是真实观测
    '''
    num_cameras = len(obs_list)
    obs_tensor = torch.zeros((num_cameras, max_visual_num, obs_dim), dtype=torch.float32, device=device)
    obs_mask = torch.zeros((num_cameras, max_visual_num), dtype=torch.float32, device=device)

    for i, cam_obs in enumerate(obs_list):
        cam_obs = np.array(cam_obs)
        if len(cam_obs) == 0:
            continue # 该相机什么都没有看到
        
        # 超出最大可视数量的目标裁剪策略
        if len(cam_obs) > max_visual_num:
            if strategy == 'nearest':
               indices = np.argsort(cam_obs[:, 0])[:max_visual_num] # 按照距离排序，取最近的max_visual_num个目标
            elif strategy == 'random':
                indices = np.random.choice(len(cam_obs), max_visual_num, replace=False) # 随机选择max_visual_num个目标
            else:
                raise ValueError("Unsupported strategy: {}".format(strategy))
            cam_obs = cam_obs[indices]
        
        obs_len = len(cam_obs)
        obs_tensor[i, :obs_len, :] = torch.tensor(cam_obs, dtype=torch.float32)
        obs_mask[i, :obs_len] = 1.0

    return obs_tensor, obs_mask

def process_obs(local_obs, num_targets, device='cpu', obs_dim=3):
    '''
    将相机的本地观测local_obs处理为张量形式，适用于Actor的输入
    local_obs (N, M*3+4+N) 3:obs_dim
    将local_obs拆分成三部分：
    -目标特征 obs_tensor[N, M, 3] 3:obs_dim
    -obs_mask[N, M] 观测掩码，表明哪些目标是真实观测
    -相机特征 camera_tensor[N, 4+N] 4:相机
    '''
    local_obs = np.array(local_obs)
    N = local_obs.shape[0]  # 相机数量
    target_feat_len = num_targets * obs_dim  # 每个相机的目标特征长度
    self_feat_len = 4 + N # 每个相机的自身特征长度（位置、角度、one-hot编码）

    obs_tensor = []
    obs_mask = []
    self_info = []
    for i in range(N):
        obs_i = local_obs[i] # [M*3+4+N]
        target_feats = obs_i[:target_feat_len].reshape(num_targets, obs_dim)  # [M, 3]
        self_feats = obs_i[target_feat_len:] # [4+N]

        # mask: 若d=-1，则表示目标不可见
        d = target_feats[:, 0]
        mask = (d != -1)
        obs_tensor.append(target_feats)
        obs_mask.append(mask)
        self_info.append(self_feats)

    obs_tensor = torch.tensor(np.array(obs_tensor), dtype=torch.float32, device=device)  # [N, M, obs_dim]
    obs_mask = torch.tensor(np.array(obs_mask), dtype=torch.bool, device=device)  # [N, M]
    self_info = torch.tensor(np.array(self_info), dtype=torch.float32, device=device)  # [N, 4+N]

    return obs_tensor, obs_mask, self_info # 返回处理后的张量形式的观测数据
