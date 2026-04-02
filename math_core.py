import numpy as np

def de_casteljau(points, t):
    """
    使用纯递归实现的 De Casteljau 算法。
    :param points: 控制点列表, 形如 [[x0,y0], [x1,y1], ...]
    :param t: 参数 t (0 <= t <= 1)
    :return: 曲线上的点 [x, y]
    """
    if len(points) == 1:
        return points[0]
    
    new_points = []
    for i in range(len(points) - 1):
        x = (1 - t) * points[i][0] + t * points[i+1][0]
        y = (1 - t) * points[i][1] + t * points[i+1][1]
        new_points.append([x, y])
        
    return de_casteljau(new_points, t)

def compute_bezier_curve(points, num_segments):
    """
    批量计算贝塞尔曲线上的点。
    """
    curve_points = np.zeros((num_segments + 1, 2), dtype=np.float32)
    for i in range(num_segments + 1):
        t = i / num_segments
        p = de_casteljau(points, t)
        curve_points[i] = p
    return curve_points

def compute_bspline_curve(points, num_segments_total):
    """
    计算均匀三次 B 样条曲线。每4个点构成一段曲线。
    利用标准的三次 B 样条基矩阵计算。
    """
    n = len(points)
    if n < 4:
        return np.zeros((0, 2), dtype=np.float32)
    
    num_curves = n - 3
    points_per_curve = num_segments_total // num_curves
    total_points = points_per_curve * num_curves + 1
    curve_points = np.zeros((total_points, 2), dtype=np.float32)
    
    # 均匀三次 B 样条的基矩阵
    M = np.array([
        [-1,  3, -3,  1],
        [ 3, -6,  3,  0],
        [-3,  0,  3,  0],
        [ 1,  4,  1,  0]
    ]) / 6.0

    idx = 0
    for i in range(num_curves):
        # 取连续的4个控制点
        P = np.array(points[i:i+4])
        
        # 为了保证曲线首尾相连，只有最后一段才计算 t=1.0 的点
        steps = points_per_curve + 1 if i == num_curves - 1 else points_per_curve
        
        for j in range(steps):
            t = j / points_per_curve
            T = np.array([t**3, t**2, t, 1])
            # T * M * P
            p = T @ M @ P
            curve_points[idx] = p
            idx += 1
            
    return curve_points[:idx]