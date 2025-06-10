import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import numpy as np

from MSMTC.DigitalPose2D.env import Pose_Env_Base

def test_goal_agent_trajectory():
    class Args:
        def __init__(self):
            self.max_steps = 1000
            self.obstacle_flag = 0
            self.target_move_type = 'goal'
            self.num_cameras = -1
            self.num_targets = -1
            self.visual_distance = -1
            self.visual_angle = -1
            self.rotation_scale = -1
            self.move_scale = -1
            self.render_save = False
            self.render = False

    args = Args()
    env = Pose_Env_Base(args)

    # 存储目标轨迹
    target_trajectories = [[] for _ in range(env.num_targets)]
    
    # 记录轨迹
    for step in range(args.max_steps):
        for i in range(env.num_targets):
            target_pos = env.targets[i][:2].copy()
            target_trajectories[i].append(target_pos)
        env.target_move()  # 让目标根据goal移动

    # 画轨迹
    plt.figure(figsize=(12, 6))
    for i, traj in enumerate(target_trajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f'Target {i}', alpha=0.7)
        plt.scatter(traj[0, 0], traj[0, 1], color='green', marker='x', s=100)  # 起点
        plt.scatter(traj[-1, 0], traj[-1, 1], color='red', marker='o', s=100)  # 终点

    plt.xlim(env.area[0], env.area[1])
    plt.ylim(env.area[2], env.area[3])
    plt.title('GoalNavAgent Targets Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.savefig('goal_agents_trajectories.png')  # 保存轨迹图
    print('轨迹图已保存为 goal_agents_trajectories.png')


    # 使用环境自带渲染（每个step保存一张）
    # env.render_save = True
    # env.render = True
    # env.reset()
    # for step in range(args.max_steps):
    #     env.target_move()
    #     env.render()  # 渲染当前状态
    # print('已保存所有步长的渲染图像到默认渲染路径')

if __name__ == "__main__":
    test_goal_agent_trajectory()
    print("测试完成，目标轨迹已绘制并保存。")