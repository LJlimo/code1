# 势能函数、覆盖评分计算函数

import numpy as np

# def calculate_potential_single(camera_pos, target_pos, visual_distance, visual_angle):
#     '''
#     计算单个相机和目标之间的势能
#     camera_pos:相机位置[x, y, angle]
#     target_pos:目标位置[x, y]
#     visual_distance:相机可视距离
#     visual_angle:相机可视角度
#     '''
#     A = 1 # 势能幅度系数
#     sigma_D =  visual_distance / 2 # 距离平滑过渡系数
#     sigma_theta = abs(visual_angle / 4) # 角度平滑过渡系数
#     beta_D =  visual_distance / 10  # 距离Sigmoid系数
#     beta_theta = abs(visual_angle / 18) # 角度Sigmoid系数

#     # 相对位置
#     dx = target_pos[0] - camera_pos[0]
#     dy = target_pos[1] - camera_pos[1]
#     distance = np.sqrt(dx**2 + dy**2) # 相机和目标之间的距离
#     angle_x = np.degrees(np.arctan2(dy, dx)) # 相机到目标向量与x轴的夹角
#     angle = angle_x - camera_pos[2] # 相机和目标之间的角度差
#     angle = abs((angle + 180) % 360 - 180) # 角度差取绝对值

#     # 计算势能函数各部分
#     distance_potential = np.tanh((visual_distance-distance) / sigma_D) # 距离势能
#     angle_potential = np.tanh((visual_angle / 2 - angle) / sigma_theta) # 角度势能

#     distance_sigmoid = 1 / (1 + np.exp(-(visual_distance - distance) / beta_D)) # 距离中心偏好项
#     angle_sigmoid = 1 / (1 + np.exp(-(visual_angle / 2 - angle) / beta_theta)) # 角度中心偏好项

#     # print(distance, angle, distance_potential, angle_potential, distance_sigmoid, angle_sigmoid)

    

#     # 计算总势能
#     U = A * distance_potential * angle_potential * distance_sigmoid * angle_sigmoid
#     if distance_potential < 0 and angle_potential < 0:
#         U = -U
#     return U

# def calculate_target_coverage_score(potential_matrix, target_id):
#     '''
#     计算单个目标的覆盖评分
#     potential_matrix:势能矩阵
#     target_id:目标id
#     '''
#     U_all = potential_matrix[:, target_id] # 获取所有相机和目标之间的势能
#     score = np.sum(np.log1p(np.exp(U_all))) # 计算覆盖评分
#     return score


def calculate_attract_potential(camera_pos, target_pos, visual_distance, visual_angle):
    '''
    计算目标对相机的吸引势能
    camera_pos:相机位置
    target_pos:目标位置
    visual_distance:相机可视距离
    visual_angle:相机可视角度
    '''
    alpha = 1 # 吸引势能幅度系数
    sigma_d = visual_distance # 距离平滑过渡系数
    sigma_theta = visual_angle / 2 # 角度平滑过渡系数

    # 相对位置
    dx = target_pos[0] - camera_pos[0]
    dy = target_pos[1] - camera_pos[1]
    distance = np.sqrt(dx**2 + dy**2) # 相机和目标之间的距离
    angle_x = np.degrees(np.arctan2(dy, dx)) # 相机到目标向量与x轴的夹角
    angle = angle_x - camera_pos[2] # 相机和目标之间的角度差
    angle = abs((angle + 180) % 360 - 180) # 角度差取绝对值

    # 计算吸引势能函数各部分
    distance_potential = np.exp((visual_distance - distance) / sigma_d) # 距离吸引势能
    angle_potential = np.exp((visual_angle / 2 - angle) / sigma_theta) # 角度吸引势能

    U_attract = alpha * distance_potential * angle_potential # 吸引势能
    return U_attract

def calculate_repel_potential(camera_pos1, camera_pos2):
    '''计算两个相机之间的斥力函数
    camera_pos1:相机1位置[x, y, angle]
    camera_pos2:相机2位置[x, y, angle]
    '''
    eta = 5.0 # 斥力幅度系数
    epsilon = 1e-3 # 防止除0错误
    delta = 1 # 防止log(0)错误

    # 相对位置(曼哈顿距离)
    manhattan_distance = np.abs(camera_pos1[0] - camera_pos2[0]) + np.abs(camera_pos1[1] - camera_pos2[1])

    # 原始斥力项
    repel_raw = eta / (manhattan_distance + epsilon) # 斥力原始值

    # log 斥力势能
    repel_log = np.log(repel_raw + delta) # 斥力势能

    return -repel_log