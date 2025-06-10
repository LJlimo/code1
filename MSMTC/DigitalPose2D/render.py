import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os

def render(area, camera_pos, target_pos, visual_distance, visual_angle, episode_num, step_num, obstacle_pos=None, visible=None, save_path=None, return_rgb_array=False, ax=None):
    '''
    area: 区域范围 [xmin, xmax, ymin, ymax]
    camera_pos: 相机位置 (x, y, angle) n个相机
    target_pos: 目标位置 (x, y) m个目标
    visual_distance: 相机可视半径
    visual_angle: 相机监控角度
    step_num: 当前步数
    obstacle_pose: 障碍物位置 (x, y, radius) 可选
    visible: 相机和目标的可见关系 n*m的矩阵 1表示可见
    save_path: 保存路径 可选
    return_rgb_array: 是否返回RGB数组, 返回内存帧直接生成gif
    ax: matplotlib轴对象, 如果提供则在该轴上绘图, 否则创建新图
    '''
    
    camera_pos = np.array(camera_pos)
    target_pos = np.array(target_pos)
    camera_num = len(camera_pos)
    target_num = len(target_pos)

    

    # ----1. 初始化画布: 外部传入ax或新建figrue----
    created_new_fig = False
    if ax is None:
        x_range = area[1] - area[0]
        y_range = area[3] - area[2]
        ratio = x_range / y_range
        short_len = 6
        if ratio > 1:
            figsize = (short_len * ratio, short_len)
        else:
            figsize = (short_len, short_len / ratio)
        fig, ax = plt.subplots(figsize=figsize)
        created_new_fig = True
    else:
        ax.clear()

    ax.set_xlim(area[0], area[1])
    ax.set_ylim(area[2], area[3])
    ax.set_aspect('equal')  # 保持像素正方形
    ax.axis('off')  # 关闭坐标轴
    for spine in ax.spines.values():  # 干净边框
        spine.set_visible(False)

    ax.set_title(f"Episode: {episode_num}_Step: {step_num}", fontsize=12, color='black', pad=15)

    # ----2. 绘制区域边界框----
    rect = patches.Rectangle((area[0], area[2]), 
                             area[1] - area[0], 
                             area[3] - area[2], 
                             linewidth=1, edgecolor='black', facecolor='none', linestyle='--')  
    ax.add_patch(rect)

    # ----3. 绘制相机位置----
    for i in range(camera_num):
        ax.plot(camera_pos[i, 0], camera_pos[i, 1], 'o', color='steelblue', markersize=3)
        ax.annotate(f'S$_{{{i+1}}}$', xy=(camera_pos[i, 0], camera_pos[i, 1]),
                    xytext=(camera_pos[i, 0] + 15, camera_pos[i, 1] + 15),
                    textcoords='data', fontsize=8, color='black')
    
    # ----4. 绘制相机可视范围虚线圆圈+透明圆----
    for i in range(camera_num):
        a, b = camera_pos[i][:2]
        theta = np.linspace(0, 2 * np.pi, 100)
        x = a + visual_distance * np.cos(theta)
        y = b + visual_distance * np.sin(theta)
        ax.plot(x, y, color='steelblue', linewidth=1, dashes=(6, 5.), dash_capstyle='round', alpha=0.9)
        disk = plt.Circle((a, b), visual_distance, color='steelblue', alpha=0.05)
        ax.add_artist(disk)
    
    # ----5. 绘制相机当前监控扇形----
    for i in range(camera_num):
        theta = camera_pos[i, 2]
        while theta < -180 or theta > 180:
            theta -= np.sign(theta) * 360
        the1 = (theta - visual_angle / 2) % 360
        the2 = (theta + visual_angle / 2) % 360
        wedge = patches.Wedge((camera_pos[i, 0], camera_pos[i, 1]),
                               visual_distance, the1, the2, color='green', alpha=0.2)
        ax.add_artist(wedge)
    
    # ----6. 绘制目标位置----
    for i in range(target_num):
        c = 'firebrick'
        for j in range(camera_num):
            if visible is not None and visible[j][i] == 1:
                c = 'yellow'
                break
        ax.plot(target_pos[i, 0], target_pos[i, 1], 'o', color=c, markersize=3)
        ax.annotate(f'T$_{{{i+1}}}$', xy=(target_pos[i, 0], target_pos[i, 1]),
                     xytext=(target_pos[i, 0] + 15, target_pos[i, 1] + 15),
                     textcoords='data', fontsize=8, color='black')

    # ----7. 绘制障碍物位置----
    if obstacle_pos is not None:
        for obs in obstacle_pos:
            disk = plt.Circle((obs[0], obs[1]), obs[2], color='grey', alpha=0.5)
            ax.add_artist(disk)


    # ----8. 保存/展示/返回----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    elif created_new_fig:
        # plt.show(block=False)  # 非阻塞显示图像
        plt.pause(0.1)  # 短暂停留以显示图像
    
    # ----9. 返回RGB数组----
    if return_rgb_array:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
        if created_new_fig:
            plt.close(fig) 
        return img
    
    # ----10. 清理和返回----
    if created_new_fig:
        plt.close(fig)  # 如果创建了新图，关闭它以释放内存
