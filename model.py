'''
本文件是5.17对多智能体模型的修改
主要是将模型的输入改为多智能体的形式
model是一个多智能体网络， 隐式处理多智能体概念，按照维度区分多智能体，整个模型接收（B, N, ...）的张量输入，B：batch_size, N:智能体数量，...:其他维度
'''

import math
import torch 
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn.init as init

from utils import norm_col_init


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = Parameter(torch.Tensor(out_features))

        #epsilon_weight 和 epsilon_bias 是用于存储噪声的缓冲区，通过 register_buffer 注册为非参数变量。这些变量不会被优化器更新，但会随模型保存和加载。
        self.register_buffer('epsilon_weight', torch.Tensor(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.Tensor(out_features))
        self.reset_parameters() # 对权重、偏置和噪声幅度进行初始化
    
    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):
            # 使用均匀分布初始化权重
            init.uniform(self.weight, -math.sqrt(3.0 / self.in_features), math.sqrt(3.0 / self.in_features))
            init.uniform(self.bias, -math.sqrt(3.0 / self.in_features), math.sqrt(3.0 / self.in_features)) # 这种初始化方式基于Xavier原则，确保权重的方差与输入特征的数量成反比，从而避免梯度爆炸或消失的问题
            init.constant(self.sigma_weight, self.sigma_init) # 使用常数值初始化噪声幅度，将其初始化为一个极小的值，可以在训练初期限制噪声的影响，逐步让模型适应噪声的存在
            init.constant(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        # 在输入数据上执行带噪声的线性变换
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, self.bias + self.sigma_bias * self.epsilon_bias)

    # 下面两个方法是动态生成和移除噪声的
    def sample_noise(self):
        self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        self.epsilon_bias = torch.randn(self.out_features)
    
    def remove_features(self):
        self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        self.epsilon_bias = torch.zeros(self.out_features)


class ActorEncoder(nn.Module):
    '''
    简单的自注意力编码器, 完成从obs_tensor到encoder_out的映射，用于Actor网络
    '''
    def __init__(self, obs_dim=2, hidden_dim=64, n_heads=2, dropout=0.1):
        super(ActorEncoder, self).__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_dim) # 输入投影层，将输入的观测维度映射到隐藏层维度
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True) # 多头自注意力层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, obs_tensor, obs_mask):
        '''
        obs_tensor: (num_cameras, max_visual_num, obs_dim) 观测张量
        obs_mask: (num_cameras, max_visual_num) 观测掩码
        返回:
        agent_feature:(N, hidden_dim) 每个Agent编码后的整体表示
        '''
        x = self.input_proj(obs_tensor) # 输入投影(N, k, obs_dim) -> (N, k, hidden_dim)

        attn_mask = ~obs_mask.bool() # 将观测掩码转换为布尔值掩码, 用于自注意力层的掩码 True表示要被mask
        attn_output, _ = self.attention(x, x, x, key_padding_mask=attn_mask) # 自注意力层(N, k, hidden_dim) -> (N, k, hidden_dim)

        # masked average pooling
        masked_output = attn_output * obs_mask.unsqueeze(-1) # 将掩码应用于自注意力输出 (N, K, hidden_dim)
        summed = masked_output.sum(dim=1) # 沿着目标数量维度求和 (N, hidden_dim)
        count = obs_mask.sum(dim=1, keepdim=True) + 1e-6 # 计算每个智能体的有效目标数量 (N, 1)
        pooled = summed / count # 计算平均值 (N, hidden_dim)

        agent_feature = self.mlp(pooled) # 通过MLP进行进一步处理 (N, hidden_dim)

        # agent_feature 是全局上下文特征（所有看到目标平均池化后）， attn_output是局部上下文特征（每个目标的自注意力输出）
        return agent_feature # 返回编码后的特征
        # 如果需要返回局部上下文特征，可以取消注释下面的行
        # return agent_feature, attn_output



class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, head_name):
        '''
        Policy网络: 这个Policy网络比较简单(只有一个线性层)的原因是，这里只是策略头，一个用于输出动作分布的网络，特征提取和学习应该在encoder层中主要完成，这样可以保持训练的稳定性。
        input_dim: 输入维度(这个input应该是前面处理后的维度)
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        head_name: 是否使用噪声头，带ns是有噪声的线性层

        '''
        super(PolicyNet, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.head_name = head_name
    
        

        if 'ns' in head_name: # 需要使用带噪声的线性层
            self.noise = True
            self.actor_linear = NoisyLinear(input_dim, action_dim, sigma_init=0.017) # NoisyLinear是一在权重和偏置中添加噪声的线形层，增强策略的探索能力，这在强化学习中对平衡探索与利用很有用 sigma_init是初始化噪声的标准差,控制噪声的强度
        else:
            self.noise = False
            self.actor_linear = nn.Linear(input_dim, action_dim)  # 普通线性层

            # 权重和偏置初始化
            self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 1) # 对权重初始化，对权重的列进行归一化，使每列的向量具有相同的标准差，用于确保模型的稳定性和效率
            self.actor_linear.bias.data.fill_(0) # 将偏置初始化为0           
    
    def forward(self, x, available_actions = None):
        # available_actions: 可用动作的掩码,用于屏蔽不可用的动作
        logits = self.actor_linear(x) # 通过线性层计算动作的logits

        if available_actions is not None:
            logits[available_actions == 0] = -1e10 # 将不可用动作的值设置为一个极小值，避免被选择

        return logits
    
    # 通过对噪声的管理，增强策略的探索能力
    def sample_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.actor_linear.remove_noise()


class ActorNet(nn.Module):
    def __init__(self, encoder:nn.Module, policy_head:nn.Module):
        '''
        ActorNet，封装了观察编码Encoder和策略头PolicyNet
        可以替换不同的编码器和策略头
        '''
        super(ActorNet, self).__init__()
        self.encoder = encoder
        self.policy_head = policy_head

    def forward(self, obs_tensor, obs_mask, availabel_actions=None):
        '''
        obs_tensor: (num_cameras, max_visual_num, obs_dim) 观测张量
        obs_mask: (num_cameras, max_visual_num) 观测掩码
        availabel_actions: (num_cameras, Action) 可用动作的掩码,用于屏蔽不可用的动作
        '''

        '''--------这里要做一个观测不到任何目标的obs的处理逻辑，现在是直接输出0向量，但可能不太好，后续想想如何改进---------'''

        num_cameras = obs_tensor.shape[0] # 获取相机数量
        hidden_dim = self.policy_head.input_dim # 获取隐藏层维度
        device = obs_tensor.device # 获取设备信息

        #检测哪些相机有有效观测
        valid_mask = (obs_mask.sum(dim=1) > 0) # (num_cameras, ) 布尔值掩码，True表示有有效观测
        features = torch.zeros((num_cameras, hidden_dim), device=device) # 初始化特征张量，形状为(num_cameras, hidden_dim)
        
        if valid_mask.any():
            features_valid = self.encoder(obs_tensor[valid_mask], obs_mask[valid_mask]) 
            features[valid_mask] = features_valid

        # features = self.encoder(obs_tensor, obs_mask) # (num_cameras, hidden_dim)
        # 进入策略头
        logits = self.policy_head(features, availabel_actions) # (num_cameras, action_dim)
        return logits

    def sample_noise(self):
        if hasattr(self.policy_head, 'sample_noise'):
            self.policy_head.sample_noise()
        
    def remove_noise(self):
        if hasattr(self.policy_head, 'remove_noise'):
            self.policy_head.remove_noise()


class CriticEncoder(nn.Module):
    '''
    针对state['global']的2层Transformer编码器（Critic用的是全局信息）
    '''
    def __init__(self, state_dim=2, hidden_dim=64, num_heads=4, out_dim=128, num_layers=2):
        '''
        state_dim: 状态中每个实体的维度，[距离，角度]
        hidden_dim:注意力编码后的维度
        num_heads:多头注意力的头数
        num_layers: Transformer编码器的层数
        '''
        super().__init__()
        self.linear_in = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        '''
        x: (batch_size, num_cameras*num_targets, state_dim) 全局状态张量
        采用这个3D形状的x的原因：1.兼容标准的Transformer和Self Attention的输入模式。2.将中间两维扁平化成序列后，更容易发现其中跨目标、跨相机的全局依赖模式
        '''
        x = self.linear_in(x) # (B, L, state_dim) -> (B, L, hidden_dim) L: num_cameras*num_targets 序列长度
        x = self.encoder(x) # (B, L, hidden_dim) -> (B, L, hidden_dim)
        x = x.mean(dim=1) # (批量大小, L, hidden_dim) -> (批量大小, hidden_dim) 对序列长度维度进行平均池化
        x = self.linear_out(x) # (批量大小, hidden_dim) -> (批量大小, out_dim)
        return x
     
class ValueNet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, head_name=None, output_dim=1):
        '''
        ValueNet: 这个网络是Critic的值函数网络,价值头—————负责从编码器输出的特征中计算状态值
        input_dim: 输入维度（编码器输出的特征维度）
        output_dim: 输出维度（值函数的输出维度） 即值
        '''
        super(ValueNet, self).__init__()
        if 'ns' in head_name:
            self.noise = True
        else:
            self.noise = False
        
        # 第一层Linear
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, std=1.0)
        self.fc1.bias.data.fill_(0)

        # 第二层Linear(输出层)
        if self.noise:
            self.fc2 = NoisyLinear(hidden_dim, output_dim, sigma_init=0.017)
        else:
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.fc2.weight.data = norm_col_init(self.fc2.weight.data, std=1.0)
            self.fc2.bias.data.fill_(0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value.squeeze(-1) # 输出形状为(batch_size, output_dim) 这里的squeeze是为了去掉最后一维
    
    def sample_noise(self):
        if self.noise:
            self.fc2.sample_noise()
    
    def remove_noise(self):
        if self.noise:
            self.fc2.remove_noise()

        
class CriticNet(nn.Module):
    def __init__(self, encoder:nn.Module, value_head:nn.Module):
        '''
        CriticNet，封装了观察编码Encoder和价值头ValueNet
        可以替换不同的编码器和价值头
        '''
        super(CriticNet, self).__init__()
        self.encoder = encoder
        self.value_head = value_head

    def forward(self, global_state_tensor):
        '''
        global_state_tensor: (batch_size, num_cameras*num_targets, state_dim) 全局状态张量
        '''
        encoded = self.encoder(global_state_tensor) 
        value = self.value_head(encoded) # shape:(B, )
        return value
    

    
        
    