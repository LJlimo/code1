"""
测试：通过 env.py 的 render() 方法调用 render1.py 的渲染效果，
并在多步交互过程中，收集每帧生成 gif。
"""

import os
import numpy as np
from MSMTC.DigitalPose2D.env import Pose_Env_Base
import imageio

def main():
    # 创建环境
    from argparse import Namespace
    args = Namespace(
        max_steps=50,  # 测试步数，短一点
        obstacle_flag=0,
        target_move_type='goal',
        num_cameras=-1,
        num_targets=-1,
        visual_distance=-1,
        visual_angle=-1,
        rotation_scale=-1,
        move_scale=-1
    )
    env = Pose_Env_Base(args)

    # 重置环境
    obs = env.reset()

    # 存储 gif 帧
    gif_frames = []

    # 交互过程
    for step in range(args.max_steps):
        # 随机动作（这里假设相机动作空间为 9 种组合）
        actions = np.random.randint(0, 9, size=env.num_cameras)

        # 执行动作并更新
        obs, rewards, done, info = env.step(actions)

        # 渲染当前帧，返回 RGB 数组
        img = env.render(return_rgb_array=True)
        gif_frames.append(img)

        print(f"Step: {step+1}/{args.max_steps}, coverage: {info['coverage_rate']:.3f}")

        if done:
            break

    # 保存 gif
    save_path = "test_env_render.gif"
    imageio.mimsave(save_path, gif_frames, duration=0.1)
    print(f"已保存交互过程 gif: {save_path}")

if __name__ == "__main__":
    main()
