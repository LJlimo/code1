from datetime import datetime
import os
import imageio
import torch
import numpy as np
from tqdm import trange
from MSMTC.DigitalPose2D.env import Pose_Env_Base
from agentnew import MAPPOAgent
from model import ActorEncoder, PolicyNet, CriticEncoder, ValueNet, ActorNet, CriticNet
from buffer import RolloutBuffer
from torch.utils.tensorboard import SummaryWriter
from utils import process_obs_list
import argparse




def train(env, agent, buffer, num_episodes, max_rollout_steps, device, args):
    pbar = trange(num_episodes, desc='Training Episodes')

    # Tensorboard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', f'coverage_logs_{current_time}')
    writer = SummaryWriter(log_dir)

    # ----一个episode代表一次完整的训练过程：采样+PPO更新----
    for episode_i in pbar: 
        
        generate_gif_every = 100  # 每100个episode生成一次gif
        generate_gif = (episode_i % generate_gif_every == 0) # 是否生成gif图像

        obs_dict = env.reset()
        obs_list = obs_dict['obs']
        global_state = obs_dict['global']  # 获取全局状态

        agent.eval() # 切换到eval, 采样阶段使用稳定的策略
        
        # ----是否要生成gif图像----
        if generate_gif:  # 每100个episode生成一次gif
            frames = [] # 用于存储每个step的图像帧

        # ----rollout采样：采样rollout_steps步----
        for step_i in range(max_rollout_steps):
            # ----将观测处理为定长（max_visual_num）转换为张量----
            obs_tensor, obs_mask = process_obs_list(obs_list, args.max_visual_num, obs_dim=env.state_dim, device=device, strategy='nearest')

            # ----与环境交互、采样动作、得到log_probs、values----
            with torch.no_grad():
                actions, log_probs = agent.select_action(obs_tensor, obs_mask, None)

            # ----与环境交互----
            next_obs_dict, rewards, dones, info = env.step(actions.tolist())
            next_obs_list = next_obs_dict['obs']
            next_global_state = next_obs_dict['global']
            if generate_gif:  # 每100个episode收集一次帧
                frame = env.render(return_rgb_array=True)  # 获取当前帧图像
                frames.append(frame)  # 存储帧图像

            # ----将当前时间步的经验存储到buffer中----
            buffer.insert(
                obs_tensor=obs_tensor,
                obs_mask=obs_mask,
                global_state=torch.tensor(global_state, dtype=torch.float32, device=device),
                actions=actions,
                log_probs=log_probs.detach().to(device),
                rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
                values=torch.zeros(env.num_cameras, device=device),  # 这里先填0，后面再计算
                dones=torch.tensor(dones, device=device)
            )

            # ----更新观测, 继续下一步----
            obs_list = next_obs_list
            global_state = next_global_state

            # ----记录覆盖率，这里记录的是采样阶段的覆盖率，未经下一次PPO训练更新参数----
            coverage = info.get('coverage_rate', None)
            visible_counts = info['visible_counts'] # (num_cameras, ) 每个相机看到的目标数量
            if coverage is not None:
                writer.add_scalar('CoverageRate/Step', coverage, global_step=episode_i * max_rollout_steps + step_i)
            for cam_id in range(env.num_cameras):
                writer.add_scalar('Camera{}/visibleTargets'.format(cam_id), visible_counts[cam_id], global_step=episode_i * max_rollout_steps + step_i)
        # =========采样阶段结束，开始PPO更新阶段==========
        # buffer中已经存储了 rollout_steps步的经验

        # ----计算Advantage（GAE）和Return----
        with torch.no_grad():
            last_values = agent.critic_net(torch.tensor(global_state, dtype=torch.float32, device=device)).squeeze(-1) #最后一步的状态值，使用Critic计算
        buffer.compute_returns_and_advantages(last_values)

        # ----切换到train模式，PPO更新----
        agent.train()

        # ----更新PPO----
        total_actor_loss, total_critic_loss, total_entropy, total_total_loss = 0, 0, 0, 0
        count = 0
        for batch in buffer.get_batches(args.batch_size, args.ppo_epochs):
            loss_dict = agent.update(batch)
            total_actor_loss += loss_dict['actor_loss']
            total_critic_loss += loss_dict['critic_loss']
            total_entropy += loss_dict['entropy']
            total_total_loss += loss_dict['total_loss']
            count += 1

        # 平均
        writer.add_scalar('Loss/Total', total_total_loss / count, global_step=episode_i)
        writer.add_scalar('Loss/Actor', total_actor_loss / count, global_step=episode_i)
        writer.add_scalar('Loss/Critic', total_critic_loss / count, global_step=episode_i)
        writer.add_scalar('Loss/Entropy', total_entropy / count, global_step=episode_i)

        # ----清空buffer 准备下一个----
        buffer.clear()

        # ----tqdm进度条更新---- 显示这里还要重新修改
        pbar.set_postfix({
            'Episode': episode_i + 1,
            'Reward': np.mean(rewards), # 本轮最后一次采样step的奖励平均值
            'Steps': max_rollout_steps # 本轮采样的步数（常数1000）
        })

        # ----定期保存模型参数----
        if args.save:
            if (episode_i + 1) % args.save_interval == 0:
                save_path = f'checkpoints/model_episode_{episode_i + 1}.pth'
                torch.save({
                    'actor_state_dict': agent.actor_net.state_dict(),
                    'critic_state_dict': agent.critic_net.state_dict(),
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'critic_optimizer': agent.critic_optimizer.state_dict(),
                }, save_path)
                print(f"Model saved to {save_path}")
        
        # ----生成gif图像----
        if generate_gif:  # 每100个episode生成一次gif
            gif_root_dir = 'output_gif'
            gif_run_dir = os.path.join(gif_root_dir, f'{current_time}')
            os.makedirs(gif_run_dir, exist_ok=True)
            gif_path = os.path.join(gif_run_dir, f'episode_{episode_i + 1}.gif')
            imageio.mimsave(gif_path, frames, duration=0.2)
            print(f'GIF saved to {gif_path}')
            frames.clear()

    writer.close()
    print("Training completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_rollout_steps', dest='max_rollout_steps', type=int, default=500, metavar='MRS', help='每个episode最大采样步数, -1表示使用配置文件中的值,默认为1000')
    parser.add_argument('--max-steps', dest='max_steps', type=int, default=1000, metavar='MS', help='每个episode最大步数, -1表示使用配置文件中的值,默认为1000')
    parser.add_argument('--obstacle-flag', dest='obstacle_flag', type=int, default=-1, metavar='OF', help='是否有障碍物, -1表示使用配置文件中的值(0) 0表示没有障碍物 1表示有障碍物')
    parser.add_argument('--target-move-type',dest='target_move_type', default='goal', metavar='TOT', help='目标移动类型, 默认goal, 可选random')
    parser.add_argument('--num-cameras', dest='num_cameras', type=int, default=6, metavar='N', help='相机数量n')
    parser.add_argument('--num-targets', dest='num_targets', type=int, default=12, metavar='M', help='目标数量m')
    parser.add_argument('--visual-distance', dest='visual_distance', type=int, default=-1, metavar='VD', help='相机可视距离, 默认为800')
    parser.add_argument('--visual-angle', dest='visual_angle', type=int, default=-1, metavar='VA', help='相机监控角度, 默认为90度')
    parser.add_argument('--rotation-scale', dest='rotation_scale', type=int, default=-1, metavar='RS', help='每个动作的旋转角度, 默认为5度')
    parser.add_argument('--move-scale', dest='move_scale', type=int, default=-1, metavar='MS', help='每个动作的位移步长, 默认为10')
    parser.add_argument('--feature-dim', dest='feature_dim', type=int, default=128, metavar='HD', help='encoder后feature的维度')
    parser.add_argument('--actor-lr', dest='actor_lr', type=float, default=0.0001, metavar='AL', help='actor的学习率')
    parser.add_argument('--critic-lr', dest='critic_lr', type=float, default=0.001, metavar='CL', help='critic的学习率')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64, metavar='BS', help='每次更新的batch size')
    parser.add_argument('--ppo-epochs', dest='ppo_epochs', type=int, default=4, metavar='PE', help='PPO的更新次数')
    parser.add_argument('--clip-eps', dest='clip_eps', type=float, default=0.2, metavar='CE', help='PPO的剪切范围')
    parser.add_argument('--entropy-coef', dest='entropy_coef', type=float, default=0.01, metavar='EC', help='熵损失的系数')
    parser.add_argument('--rollout-steps', dest='rollout_steps', type=int, default=128, metavar='RS', help='rollout的步数')
    parser.add_argument('--save', dest='save', action='store_true', help='是否保存模型')
    parser.add_argument('--save-interval', dest='save_interval', type=int, default=100, metavar='SI', help='保存模型的间隔')
    parser.add_argument('--model', dest='model', default='model', metavar='M', help='使用的模型名称（是否带噪音）')
    parser.add_argument('--max-visual-num', dest='max_visual_num', type=int, default=5, metavar='MVN', help='最大观测长度')

    args = parser.parse_args()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 1.环境初始化
    env = Pose_Env_Base(args)    

    # 2.各网络模块初始化
    actor_encoder = ActorEncoder(obs_dim=env.state_dim, hidden_dim=args.feature_dim)
    critic_encoder = CriticEncoder(state_dim=env.state_dim, hidden_dim=args.feature_dim)
    policy_net = PolicyNet(input_dim=args.feature_dim, action_dim=9, hidden_dim=512, head_name=args.model)
    value_net = ValueNet(input_dim=args.feature_dim, hidden_dim=512, head_name=args.model)
    actor_net = ActorNet(actor_encoder, policy_net)
    critic_net = CriticNet(critic_encoder, value_net)

    # 3.Agent 初始化
    agent = MAPPOAgent(actor_net, critic_net, device)

    # 4.Buffer初始化
    buffer = RolloutBuffer(
        rollout_steps=args.rollout_steps,
        num_cameras=env.num_cameras,
        num_targets=env.num_targets,
        obs_shape=(args.max_visual_num, env.state_dim),
        global_state_shape=(env.num_cameras, env.num_targets, env.state_dim),
        device=device
    )

    train(env, agent, buffer, num_episodes=1000, max_rollout_steps=args.max_rollout_steps, device=device, args=args)



        