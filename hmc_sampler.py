import numpy as np

def leapfrog(q, p, grad_log_prob, path_len, step_size):
    """
    呱！
    
    Args:
        q (np.ndarray): 当前位置
        p (np.ndarray): 当前动量
        grad_log_prob (function): 目标分布对数概率的梯度函数
        path_len (float): 积分路径长度
        step_size (float): 每一步的步长
        
    Returns:
        np.ndarray, np.ndarray: 积分结束后的新位置和新动量
    """
    q_new, p_new = q.copy(), p.copy()
    num_steps = int(path_len / step_size)
    
    # U(q) = -log_prob(q), F = -grad(U) = grad_log_prob(q)
    
    # 半步更新动量
    p_new += 0.5 * step_size * grad_log_prob(q_new)
    
    for _ in range(num_steps - 1):
        # 整步更新位置
        q_new += step_size * p_new
        # 整步更新动量
        p_new += step_size * grad_log_prob(q_new)
        
    # 整步更新位置
    q_new += step_size * p_new
    # 最终半步更新动量
    p_new += 0.5 * step_size * grad_log_prob(q_new)
    
    return q_new, -p_new  # 返回反向动量以保持细节平衡

def hamiltonian_monte_carlo(target_dist, n_samples, burn_in, path_len=1.0, step_size=0.1):
    """
    Args:
        target_dist: 目标分布对象，需要 .log_prob() 和 .grad_log_prob() 方法
        n_samples (int): 需要生成的样本数量
        burn_in (int): 老化期
        path_len (float): 蛙跳积分的路径长度
        step_size (float): 蛙跳积分的步长
        
    Returns:
        np.ndarray: 采样点数组
        np.ndarray: 接受率历史
    """
    current_q = np.array([0.0, 0.0])
    samples = []
    acceptance_history = []
    accepted_count = 0
    
    total_steps = n_samples + burn_in
    for i in range(total_steps):
        # 从高斯分布中采样初始动量 p
        p0 = np.random.normal(0, 1, size=2)
        
        # 使用蛙跳法模拟轨迹，得到提议点 q_new
        q_new, p_new = leapfrog(current_q, p0, target_dist.grad_log_prob, path_len, step_size)
        
        # 计算接受概率
        # H(q, p) = U(q) + K(p) = -log_prob(q) + 0.5 * p^T p
        current_U = -target_dist.log_prob(current_q)
        current_K = 0.5 * np.sum(p0**2)
        
        proposal_U = -target_dist.log_prob(q_new)
        proposal_K = 0.5 * np.sum(p_new**2)
        
        log_alpha = (current_U + current_K) - (proposal_U + proposal_K)
        alpha = np.exp(min(0, log_alpha))
        
        if np.random.rand() < alpha:
            current_q = q_new
            if i >= burn_in:
                accepted_count += 1
                
        if i >= burn_in:
            samples.append(current_q.copy())
            acceptance_history.append(accepted_count / (i - burn_in + 1))

    return np.array(samples), np.array(acceptance_history)
