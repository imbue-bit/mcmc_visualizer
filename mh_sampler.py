import numpy as np

def metropolis_hastings(target_dist, n_samples, burn_in, step_size=1.0):
    """
    使用随机游走 Metropolis-Hastings 算法采样
    
    Args:
        target_dist: 目标分布对象，需要有 .log_prob() 方法
        n_samples (int): 需要生成的样本数量
        burn_in (int): 老化期，丢弃的初始样本数量
        step_size (float): 提议分布（高斯）的标准差
        
    Returns:
        np.ndarray: 采样点数组
        np.ndarray: 接受率历史
    """
    # 初始点
    current_pos = np.array([0.0, 0.0])
    samples = []
    acceptance_history = []
    accepted_count = 0
    
    total_steps = n_samples + burn_in
    for i in range(total_steps):
        # 提议一个新点
        proposal_pos = current_pos + np.random.normal(0, step_size, size=2)
        
        # 算alpha
        log_prob_current = target_dist.log_prob(current_pos)
        log_prob_proposal = target_dist.log_prob(proposal_pos)
        
        log_alpha = log_prob_proposal - log_prob_current
        alpha = np.exp(min(0, log_alpha))
        
        if np.random.rand() < alpha:
            current_pos = proposal_pos
            if i >= burn_in:
                accepted_count += 1
        
        if i >= burn_in:
            samples.append(current_pos.copy())
            acceptance_history.append(accepted_count / (i - burn_in + 1))

    return np.array(samples), np.array(acceptance_history)
