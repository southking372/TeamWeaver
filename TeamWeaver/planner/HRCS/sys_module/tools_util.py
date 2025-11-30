import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def clamp(x, m, M):
    return max(min(x, M), m)

def get_task_assignment(alpha):
    """
    根据alpha矩阵确定任务分配，与MATLAB版本保持一致
    """
    # 找出每个机器人分配的任务（最大alpha值对应的任务）
    task_assignment = np.argmax(alpha, axis=0) + 1  # +1转为1-based索引
    
    # 检查是否有机器人未分配任务（所有alpha值接近0）
    for i in range(alpha.shape[1]):
        if np.max(alpha[:, i]) < 1e-3:  # 使用max而不是norm
            task_assignment[i] = 0  # 使用0表示未分配任务
    
    return task_assignment

def print_info(t, time_to_synthesize_controller, task_assignment, u, delta, S):
    print("Time:", t)
    print("Time to synthesize controller:", time_to_synthesize_controller)
    print("Task assignment:", task_assignment)
    print("Input:\n", u)
    print("Delta:\n", delta)
    print("Specialization:\n", S)

def plot_quad(h_quad, x_traj):
    x = x_traj[0, 4, -1]
    y = x_traj[1, 4, -1]
    
    # Calculate direction
    dir_x = x_traj[0, 4, -1] - x_traj[0, 4, -2]
    dir_y = x_traj[1, 4, -1] - x_traj[1, 4, -2]
    
    th = np.arctan2(dir_y, dir_x) + np.pi/4
    l = 0.18
    r = 0.04
    
    # Calculate quadrotor corners
    q1 = np.array([x, y]) + l/2 * np.array([np.cos(th), np.sin(th)])
    q2 = np.array([x, y]) + l/2 * np.array([np.cos(th+np.pi/2), np.sin(th+np.pi/2)])
    q3 = np.array([x, y]) + l/2 * np.array([np.cos(th+np.pi), np.sin(th+np.pi)])
    q4 = np.array([x, y]) + l/2 * np.array([np.cos(th+3*np.pi/2), np.sin(th+3*np.pi/2)])
    
    # Calculate propeller positions
    theta = np.linspace(0, 2*np.pi, 36)
    prop_x = r * np.cos(theta)
    prop_y = r * np.sin(theta)
    
    if not h_quad:
        h_quad = [None] * 6
        h_quad[0], = plt.plot([q1[0], q3[0]], [q1[1], q3[1]], 'k-', linewidth=2)
        h_quad[1], = plt.plot([q2[0], q4[0]], [q2[1], q4[1]], 'k-', linewidth=2)
        
        # Propellers
        prop1_x = q1[0] + prop_x
        prop1_y = q1[1] + prop_y
        h_quad[2] = plt.fill(prop1_x, prop1_y, color=[0.5, 0.5, 0.5], edgecolor='k', linewidth=2, alpha=0.5)[0]
        
        prop2_x = q2[0] + prop_x
        prop2_y = q2[1] + prop_y
        h_quad[3] = plt.fill(prop2_x, prop2_y, color=[0.5, 0.5, 0.5], edgecolor='k', linewidth=2, alpha=0.5)[0]
        
        prop3_x = q3[0] + prop_x
        prop3_y = q3[1] + prop_y
        h_quad[4] = plt.fill(prop3_x, prop3_y, color=[0.5, 0.5, 0.5], edgecolor='k', linewidth=2, alpha=0.5)[0]
        
        prop4_x = q4[0] + prop_x
        prop4_y = q4[1] + prop_y
        h_quad[5] = plt.fill(prop4_x, prop4_y, color=[0.5, 0.5, 0.5], edgecolor='k', linewidth=2, alpha=0.5)[0]
    else:
        h_quad[0].set_data([q1[0], q3[0]], [q1[1], q3[1]])
        h_quad[1].set_data([q2[0], q4[0]], [q2[1], q4[1]])
        
        h_quad[2].set_xy(np.column_stack((q1[0] + prop_x, q1[1] + prop_y)))
        h_quad[3].set_xy(np.column_stack((q2[0] + prop_x, q2[1] + prop_y)))
        h_quad[4].set_xy(np.column_stack((q3[0] + prop_x, q3[1] + prop_y)))
        h_quad[5].set_xy(np.column_stack((q4[0] + prop_x, q4[1] + prop_y)))
        
    return h_quad

def plot_fov(h_fov, x, task_assignment):
    idx_task_2 = np.where(task_assignment == 2)[0]  # 执行任务2的机器人索引
    if len(idx_task_2) == 0:
        return h_fov
        
    xy_robot = x[0:2, idx_task_2]
    th_cam = x[2, idx_task_2]
    L = 100
    # 修正视场角度为MATLAB中的值
    TH = np.pi/72  # 使用与MATLAB相同的视场角
    
    if not h_fov:
        h_fov = [None] * len(idx_task_2)
        for i in range(len(idx_task_2)):
            fov_x = xy_robot[0, i] + np.array([0, L*np.cos(th_cam[i]-TH), L*np.cos(th_cam[i]+TH)])
            fov_y = xy_robot[1, i] + np.array([0, L*np.sin(th_cam[i]-TH), L*np.sin(th_cam[i]+TH)])
            h_fov[i] = plt.fill(fov_x, fov_y, color=[1, 1, 0], alpha=0.2, edgecolor='none')[0]
    else:
        for i in range(min(len(idx_task_2), len(h_fov))):
            fov_x = xy_robot[0, i] + np.array([0, L*np.cos(th_cam[i]-TH), L*np.cos(th_cam[i]+TH)])
            fov_y = xy_robot[1, i] + np.array([0, L*np.sin(th_cam[i]-TH), L*np.sin(th_cam[i]+TH)])
            h_fov[i].set_xy(np.column_stack((fov_x, fov_y)))
            
    return h_fov

def read_mud_file(filename='mud.txt'):
    x_mud_data = []
    y_mud_data = []
    current_array = None
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()        
        for line in lines:
            line = line.strip()
            if 'x_mud:' in line:
                current_array = x_mud_data
                continue
            elif 'y_mud:' in line:
                current_array = y_mud_data
                continue
            if current_array is not None and line:
                # 处理每行数据，去除逗号和多余的空格
                values = line.replace(',', ' ').split()
                for val in values:
                    if val and val.replace('.', '', 1).replace('-', '', 1).isdigit():
                        current_array.append(float(val))
        return np.array(x_mud_data), np.array(y_mud_data)
    
    except Exception as e:
        print(f"读取mud文件出错: {e}")
        # 返回默认的简单mud区域作为备用
        return np.array([-0.5, 0.5, 0.5, -0.5, -0.5]), np.array([-0.5, -0.5, 0.5, 0.5, -0.5])

def x_perimeter_ring(t, p_transport_t):
    theta1 = np.linspace(0, 2*np.pi, 36)
    theta2 = np.linspace(2*np.pi, 0, 36)
    return p_transport_t[0] + np.concatenate([0.3*np.cos(theta1), 0.5*np.cos(theta2)])

def y_perimeter_ring(t, p_transport_t):
    theta1 = np.linspace(0, 2*np.pi, 36)
    theta2 = np.linspace(2*np.pi, 0, 36)
    return p_transport_t[1] + np.concatenate([0.3*np.sin(theta1), 0.5*np.sin(theta2)])