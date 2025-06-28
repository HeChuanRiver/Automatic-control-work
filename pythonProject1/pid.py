import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from deap import base, creator, tools, algorithms
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 系统参数（基于第一部分辨识结果）
K = 9.74  # 增益 (°C/V)
T = 2966.52  # 时间常数 (秒)
L = 209.50  # 时滞 (秒)
setpoint = 35.0  # 目标温度 (°C)
y_initial = 16.85  # 初始温度 (°C)


# 系统模型（一阶加时滞）
def system_model(y, t, u, t_history, u_history):
    if t < L:
        u_delayed = 0.0
    else:
        u_delayed = np.interp(t - L, t_history, u_history)
    dydt = (-y + K * u_delayed) / T
    return dydt


# PID控制器
def pid_controller(error, t, t_history, error_history, Kp, Ti, Td):
    if len(t_history) < 2:
        return Kp * error
    dt = t_history[-1] - t_history[-2]
    integral = np.trapz(error_history, t_history)
    if dt == 0:
        derivative = 0
    else:
        derivative = (error - error_history[-1]) / dt
    return Kp * (error + integral / Ti + Td * derivative)


# 闭环系统仿真（延长模拟时间）
def simulate_pid(params, t_max=12000, dt=10.0):
    Kp, Ti, Td = params
    t = np.arange(0, t_max, dt)
    y = np.zeros_like(t)
    u = np.zeros_like(t)
    y[0] = y_initial
    error_history = []
    t_history = []
    u_history = []

    for i in range(1, len(t)):
        error = setpoint - y[i - 1]  # 直接误差计算
        t_history.append(t[i - 1])
        error_history.append(error)
        u[i] = pid_controller(error, t[i], t_history, error_history, Kp, Ti, Td)
        u[i] = np.clip(u[i], 0, 3.5)  # 控制信号限幅（0-3.5V）
        u_history.append(u[i])
        y[i] = y[i - 1] + dt * system_model(y[i - 1], t[i], u[i], t_history, u_history)

    return t, y, u


# 目标函数（综合时间误差ITE）
def objective_function(params):
    t, y, _ = simulate_pid(params)
    error = np.abs(setpoint - y)
    ite = np.trapz(error * t, t)
    return (ite,)


# 遗传算法优化（初始化为Cohen-Coon参数附近）
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Cohen-Coon参数作为初始范围中心
toolbox.register("attr_kp", random.uniform, 1.0, 3.0)  # 围绕1.916
toolbox.register("attr_ti", random.uniform, 400, 600)  # 围绕500.86
toolbox.register("attr_td", random.uniform, 50, 100)  # 围绕75.22
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_kp, toolbox.attr_ti, toolbox.attr_td), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", objective_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=[0.5, 50, 10], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
pop = toolbox.population(n=30)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, stats=stats, halloffame=hof, verbose=True)

# 最佳PID参数
best_params = hof[0]
print(f"最佳PID参数: Kp={best_params[0]:.4f}, Ti={best_params[1]:.4f}, Td={best_params[2]:.4f}")

# 仿真最佳参数的响应
t, y, u = simulate_pid(best_params)

# 计算动态和稳态指标
y_scaled = y
error = setpoint - y_scaled
steady_state_idx = int(0.8 * len(t))
steady_state_error = np.mean(np.abs(error[steady_state_idx:]))
step_change = setpoint - y[0]
level_10 = y[0] + 0.1 * step_change
level_90 = y[0] + 0.9 * step_change
idx_10 = np.where(y_scaled >= level_10)[0][0] if np.any(y_scaled >= level_10) else len(t) - 1
idx_90 = np.where(y_scaled >= level_90)[0][0] if np.any(y_scaled >= level_90) else len(t) - 1
rise_time = t[idx_90] - t[idx_10]
overshoot = (np.max(y_scaled) - setpoint) / setpoint * 100 if np.max(y_scaled) > setpoint else 0
settling_time = t[np.where(np.abs(error[steady_state_idx:]) > 0.02 * setpoint)[0][-1] + steady_state_idx] if np.any(
    np.abs(error[steady_state_idx:]) > 0.02 * setpoint) else t[-1]

print(f"动态指标:")
print(f"上升时间: {rise_time:.2f} 秒")
print(f"超调量: {overshoot:.2f}%")
print(f"调节时间: {settling_time:.2f} 秒")
print(f"稳态误差: {steady_state_error:.4f} °C")

# 绘制结果
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, y_scaled, 'b-', label='温度 (°C)')
plt.axhline(setpoint, color='r', linestyle='--', label='设定点 35°C')
plt.xlabel('时间 (秒)')
plt.ylabel('温度 (°C)')
plt.title('闭环温度响应')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, u, 'g-', label='控制信号 (V)')
plt.xlabel('时间 (秒)')
plt.ylabel('控制信号 (V)')
plt.title('PID控制信号')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('pid_response_corrected.png')
plt.show()