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
        # 记录原始形状，以便最后恢复
        original_shape = theta.shape[:-1]

        x = theta[..., 0]
        y = theta[..., 1]
        
        y_prime = y - self.b * (x**2 - self.a**2)
        x_prime = x / self.a
        
        theta_prime = np.stack([x_prime, y_prime], axis=-1)
        
        num_points = np.prod(original_shape) if original_shape else 1
        theta_prime_reshaped = theta_prime.reshape((num_points, 2))
        
        log_pdf_flat = self.norm.logpdf(theta_prime_reshaped)
        
        # 将结果重塑回原始形状
        return log_pdf_flat.reshape(original_shape)

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
