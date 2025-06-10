Cam_Pose = [[-742, 706, -62.842588558231455], [-843, 69, -26.590794532324466], [510, 703, -135.84636503921902],
            [466, -609, 153.13035548432399]] # x, y, angle
Target_Pose = [[473.90650859, -666.624028],
               [-64.83188835, -233.64760113],
               [-980.29575616, 201.18355808],
               [-493.24174167, 655.69319226],
               [-571.57383471, -673.35637078]]
Target_camera_dict = {0: [], 1: [3], 2: [], 3: [], 4: [1]}
Camera_target_dict = {0: [], 1: [4], 2: [], 3: [1]}
Distance = [[1829.24686786, 1158.22893495, 558.23338079, 253.79410157, 1389.84498251],
            [1477.15002847, 834.94980715, 190.58493563, 683.03714475, 790.42086539],
            [1423.29036457, 1098.97244214, 1572.51428681, 1004.35647371, 1750.47388421],
            [122.3020243, 650.13223042, 1657.7601793, 1587.3227742, 1039.56779718]]
# reward = [0.4]
goals4cam = [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0]]
comm_edges = [[0, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0]]

import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

# def render(camera_pos, target_pos, obstacle_pos=None, goal=None,
#             comm_edges=None, obstacle_radius=None, save=False, visible=None): # obstacle_radius: 可以不要，放在obstacle_pos中

def render(area, camera_pos, target_pos, visual_distance, visual_angle, step_num, obstacle_pos=None, goal=None,
            comm_edges=None, save=False, visible=None):
    '''
    length: 1000 设定区域大小为2000*2000（m）x:-1000-1000 y:-1000-1000
    visual_distance: 相机可视半径为100（m） 
    相机角度: 从-180度到180度
    监控范围: 90度
    camera_pos: 相机位置 (x, y, angle) n个相机
    target_pos: 目标位置 (x, y) m个目标
    obstacle_pos: 障碍物位置 (x, y, radius)
    goal: 是当前传感器追踪的目标
    comm_edges: 通信边 n*n的矩阵(相机之间的通信) 1表示有通信
    obstacle_radius: 障碍物半径
    save: 是否保存图片
    visible: 相机和目标的可见关系 n*m的矩阵 1表示可见 可以传入，后面考虑要不要改成计算
    '''
    # 转换位置数据
    camera_pos = np.array(camera_pos)
    target_pos = np.array(target_pos)

    camera_num = len(camera_pos)
    target_num = len(target_pos)

    

    #初始化画布
    # length = 2000  # 区域大小
    img = np.zeros((area[1] - area[0] + 1, area[3] - area[2] + 1, 3)) + 255
    fig = plt.figure(0)
    plt.imshow(img.astype(np.uint8))
    plt.cla()

    ax = plt.gca() # 获取当前坐标轴
    
    #取消边框和轴线
    for spine in ax.spines.values(): 
        spine.set_visible(False)
    plt.axis('off')
    
    # 设置坐标轴范围
    plt.xlim(area[0], area[1]) 
    plt.ylim(area[2], area[3])
    
    # 绘制矩形边界框
    boundary_xmin, boundary_xmax = area[0], area[1]
    boundary_ymin, boundary_ymax = area[2], area[3]
    boundary = Rectangle((boundary_xmin, boundary_ymin),
                         boundary_xmax - boundary_xmin,
                         boundary_ymax - boundary_ymin,
                         linewidth=2, edgecolor='black',
                         facecolor='none', linestyle='--')
    ax.add_patch(boundary)
    
    # 调整坐标轴范围以显示超出部分
    margin = 200
    plt.xlim(boundary_xmin - margin, boundary_xmax + margin)
    plt.ylim(boundary_ymin - margin, boundary_ymax + margin)

    # 绘制相机位置
    for i in range(camera_num):
        plt.plot(camera_pos[i][0], camera_pos[i][1], 'o', color='steelblue', markersize=3)
        plt.annotate(f'S$_{{{i+1}}}$', xy=(camera_pos[i][0], camera_pos[i][1]), 
                     xytext=(camera_pos[i][0] + 15, camera_pos[i][1] + 15), 
                     textcoords='data', fontsize=8, color='black')

    # 绘制相机可视范围，即相机虚线圆圈
    for i in range(camera_num):
        a, b = camera_pos[i][:2]
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = a + visual_distance * np.cos(theta)
        y = b + visual_distance * np.sin(theta)
        plt.plot(x, y, linewidth=1, color='steelblue', dashes=(6, 5.), dash_capstyle='round', alpha=0.9)

        # 填充圆
        disk_camera = plt.Circle((camera_pos[i][0], camera_pos[i][1]), visual_distance, color='steelblue', fill=True, alpha=0.05)
        ax.add_artist(disk_camera)
    
    # 绘制相机当前监控区域
    for i in range(camera_num):
        theta = camera_pos[i][2] # 相机当前角度
        while theta < -180 or theta > 180:
            if theta < -180:
                theta += 360
            elif theta > 180:
                theta -= 360
        the1 = theta - visual_angle/2 % 180 # 监控区域起始角度 取180余数，防止角度超出范围
        the2 = theta + visual_angle/2 % 180 # 监控区域终止角度
        a, b = camera_pos[i][:2]
        wedge = patches.Wedge((a, b), visual_distance, the1, the2, color='green', alpha=0.2) # 绘制相机当前监控扇形
        ax.add_artist(wedge)

    # 绘制目标位置
    for i in range(target_num):
        c = 'firebrick' # 目标可见是红色
        for j in range(camera_num):
            if visible is not None and visible[j][i] == 1:
                c = 'yellow'

        plt.plot(target_pos[i][0], target_pos[i][1], 'o', color=c, markersize=3)
        plt.annotate(f'T$_{{{i+1}}}$', xy=(target_pos[i][0], target_pos[i][1]),
                     xytext=(target_pos[i][0] + 15, target_pos[i][1] + 15),
                     textcoords='data', fontsize=8, color='black')

    # 绘制障碍物
    if obstacle_pos is not None:
        obstacle_num = len(obstacle_pos)
        for i in range(obstacle_num):
            disk_obstacle = plt.Circle((obstacle_pos[i][0], obstacle_pos[i][1]), obstacle_pos[i][2], color='grey', fill=True)
            ax.add_artist(disk_obstacle)

    # 绘制通信边
    if comm_edges is not None:
        sum_edges = np.sum(comm_edges)
        for i in range(camera_num):
            for j in range(camera_num):
                if comm_edges[i][j] == 1:
                    dx, dy = camera_pos[j][0] - camera_pos[i][0], camera_pos[j][1] - camera_pos[i][1]
                    ax.arrow(camera_pos[i][0], camera_pos[i][1], dx, dy, head_width=50, head_length=70, length_includes_head=True, ec='skyblue', alpha=0.6)
            # plt.text(600, 500 + i * 30, str(comm_edges[i]))
        # plt.text(800, 800, f'Total {sum_edges} Comm Edges', fontsize=10, color='black')


    plt.show()

    





    # area_length = 1 
    # target_pos[:, :2] = (target_pos[:, :2] + 1) / 2
    # camera_pos[:, :2] = (camera_pos[:, :2] + 1) / 2
    # abs_angles = [camera_pos[i][2] - 1 for i in range(len(camera_pos))]
    # print(abs_angles)

    

    if save:
        save_dir = 'output/test1'
        plt.savefig(f"{save_dir}/step{step_num:04d}.png", bbox_inches='tight')
    plt.pause(0.01) # plt.pause()暂停0.01秒


if __name__ == '__main__':
    # render(Cam_Pose, Target_Pose, None, None, comm_edges, reward, np.array(goals4cam))
    render(Cam_Pose, Target_Pose, None, None, comm_edges, False, goals4cam)