import numpy as np
from scipy.stats import multivariate_normal

class BananaDistribution:
    def __init__(self, b=0.5, a=2.0, mean=None, cov=None):
        """
        Args:
            b (float): 控制香蕉的弯曲程度
            a (float): 控制x轴的缩放
            mean (list): 原始高斯分布的均值
            cov (list): 原始高斯分布的协方差矩阵
        """
        self.b = b
        self.a = a
        if mean is None:
            mean = [0, 0]
        if cov is None:
            cov = [[1, 0], [0, 1]]
        
        self.norm = multivariate_normal(mean=mean, cov=cov)

    def log_prob(self, theta):
        """
        计算给定点 theta = [x, y] 的对数概率密度
        这是未归一化的，但对于MCMC算法已经足够
        
        Args:
            theta (np.ndarray): 一个二维点 [x, y]
            
        Returns:
            float: 对数概率密度值
        """
        x = theta[..., 0]
        y = theta[..., 1]
        # 这是非线性变换的反变换
        y_prime = y - self.b * (x**2 - self.a**2)
        x_prime = x / self.a
        
        # 变换后的坐标
        theta_prime = np.array([x_prime, y_prime])
        
        # 计算原始高斯分布的对数概率
        return self.norm.logpdf(theta_prime)

    def grad_log_prob(self, theta):
        """
        Args:
            theta (np.ndarray): 一个二维点 [x, y]。
            
        Returns:
            np.ndarray: 梯度向量 [d/dx, d/dy]。
        """
        x, y = theta
        b = self.b
        a = self.a

        # 反变换
        x_prime = x / a
        y_prime = y - b * (x**2 - a**2)

        # 原始高斯分布的梯度 (-inv(cov) @ (theta_prime - mean))
        # 假设 mean=[0,0], cov=I, 则梯度为 [-x_prime, -y_prime]
        grad_x_prime = -x_prime
        grad_y_prime = -y_prime
        
        # 链式法则
        # d(logP)/dx = d(logP)/dx' * dx'/dx + d(logP)/dy' * dy'/dx
        # d(logP)/dy = d(logP)/dx' * dx'/dy + d(logP)/dy' * dy'/dy
        grad_x = grad_x_prime * (1/a) + grad_y_prime * (-2 * b * x)
        grad_y = grad_y_prime * 1

        return np.array([grad_x, grad_y])
