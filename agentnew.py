import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class MAPPOAgent:
    def __init__(self, actor_net, critic_net, device='cpu', actor_lr=1e-4, critic_lr=1e-3, clip_eps=0.2, entropy_coef=0.01, critic_coef=0.5):
        self.device = device
        self.actor_net = actor_net.to(device)
        self.critic_net = critic_net.to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)

        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef

    
    def select_action(self, obs_tensor, obs_mask, avail_actions=None):
        '''
        根据观察选择动作
        输入：
            obs_tensor:(N, max_visual_num, obs_dim) 观测张量
            obs_mask: 观测掩码 (N, max_visual_num)
            availabel_actions: 可用动作的掩码,用于屏蔽不可用的动作
        输出：
            actions: (N,)
            log_probs: (N,)
        '''
        logits = self.actor_net(obs_tensor, obs_mask, avail_actions) # (N, action_dim)
        probs = F.softmax(logits, dim=-1) # (N, action_dim)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions) # (N,) 采用动作的对数概率

        return actions, log_probs
    
    def evaluate_actions(self, obs_tensor, obs_mask, actions, avail_actions, global_state):
        '''
        在训练时使用：在当前新的策略下重新计算新的log_probs和熵，后面用作策略更新
        输入：
            obs_tensor:(N, max_visual_num, obs_dim) 观测张量
            obs_mask: 观测掩码 (N, max_visual_num)
            actions: (N,)
            avail_actions: 可用动作的掩码,用于屏蔽不可用的动作
            global_state: 全局状态
        输出：
            log_probs: (N,)
            entropy: (N,)
        '''
        logits = self.actor_net(obs_tensor, obs_mask, avail_actions)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions) # (N,)
        entropy = dist.entropy()

        # Critic
        values = self.critic_net(global_state).squeeze(-1) # (N,)
        return log_probs, entropy, values
    
    def compute_ppo_loss(self, obs_tensor, obs_mask, global_state, actions, old_log_probs, advantages, returns, clip_eps=0.2, entropy_coef=0.01, critic_coef=0.5):
        # ----Actor 前向计算----
        logits = self.actor_net(obs_tensor, obs_mask) # 输出logits （B, action_dim)
        dist = torch.distributions.Categorical(logits=logits) # 创建分布对象
        new_log_probs = dist.log_prob(actions) # 计算新的log_probs, (B,)
        entropy = dist.entropy().mean() # 计算熵, (B,)

        # ----比例 r----
        ratio = torch.exp(new_log_probs - old_log_probs) # (B,)

        # ----PPO clip 策略损失（Actor）----
        surr1 = ratio * advantages # (B,)
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages # (B,)
        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy # 这里的actor损失加上了熵损失，便于后面通过actor的优化器更新(B,)

        # ----Critic 前向计算----
        values = self.critic_net(global_state).squeeze(-1) # (B,)
        critic_loss = F.mse_loss(values, returns) # (B,)

        # ----总损失, 仅用于日志----
        total_loss = actor_loss + critic_coef * critic_loss # (B,)

        # ----日志记录----
        loss_dict = {
            'total_loss': total_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
        }

        return actor_loss, critic_loss, entropy, loss_dict

    def update(self, batch):
        '''
        使用PPO方式更新Actor和Critic
        输入：
            batch: (obs, obs_mask, state, actions, log_probs, advantages, returns)
            global_states: 全局状态
        '''
        obs_tensor, obs_mask, global_state, actions, old_log_probs, advantages, returns = batch

        # 计算损失
        actor_loss, critic_loss, entropy, loss_dict = self.compute_ppo_loss(
            obs_tensor=obs_tensor,
            obs_mask=obs_mask,
            global_state=global_state,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
        )

        # ----分开更新 Actor ----
        self.actor_optimizer.zero_grad()
        # actor_loss.backward(retain_graph=True)
        actor_loss.backward() # 不共享encoder可以不用这个参数
        self.actor_optimizer.step()

        # ----分开更新 Critic ----
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return loss_dict
    
    def train(self):
        self.actor_net.train()
        self.critic_net.train()
    
    def eval(self):
        self.actor_net.eval()
        self.critic_net.eval()

    def sample_noise(self):
        if hasattr(self.actor_net, 'sample_noise'):
            self.actor_net.sample_noise()
        if hasattr(self.critic_net.value_head, 'sample_noise'):
            self.critic_net.value_head.sample_noise()

    def remove_noise(self):
        if hasattr(self.actor_net, 'remove_noise'):
            self.actor_net.remove_noise()
        if hasattr(self.critic_net.value_head, 'remove_noise'):
            self.critic_net.value_head.remove_noise()


