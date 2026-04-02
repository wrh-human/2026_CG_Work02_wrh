import taichi as ti
import numpy as np
from math_core import compute_bezier_curve, compute_bspline_curve

ti.init(arch=ti.gpu)

# --- 常量定义 ---
RES = 800
# 巨型像素尺寸：20 代表每个“逻辑大像素”占据 20x20 个物理像素
# 这样锯齿感会极其强烈，肉眼可见巨大的阶梯块
PIXEL_SIZE = 20 
LOGICAL_RES = RES // PIXEL_SIZE

NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# 莫兰迪配色 (使用 tuple 格式以兼容 Taichi GUI)
BG_COLOR_V = (0.92, 0.90, 0.88)
BEZIER_COLOR = (0.50, 0.60, 0.53) # 莫兰迪绿
BSPLINE_COLOR = (0.45, 0.55, 0.65) # 莫兰迪蓝
LINE_COLOR = (0.7, 0.7, 0.7)

# --- GPU 显存缓冲区 ---
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.math.vec3(0.92, 0.90, 0.88)

@ti.kernel
def draw_pixelated_kernel(n: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
    """
    【按下A键激活】极端光栅化：模拟原始低分辨率显示效果。
    原理：将浮点坐标强行截断到巨大的逻辑像素网格中，产生巨型锯齿。
    """
    color = ti.math.vec3(r, g, b)
    for i in range(n):
        p = curve_points_field[i]
        # 核心逻辑：映射到低分辨率网格 [0, 40] 之间
        lx = ti.cast(p[0] * LOGICAL_RES, ti.i32)
        ly = ti.cast(p[1] * LOGICAL_RES, ti.i32)
        
        # 填充对应的 20x20 物理像素区域
        for dx, dy in ti.ndrange(PIXEL_SIZE, PIXEL_SIZE):
            px, py = lx * PIXEL_SIZE + dx, ly * PIXEL_SIZE + dy
            if 0 <= px < RES and 0 <= py < RES:
                pixels[px, py] = color

def main():
    window = ti.ui.Window("Graphics Lab: Smooth vs Extreme Aliasing", (RES, RES))
    canvas = window.get_canvas()
    
    control_points = []
    show_pixelated = False # 默认关闭像素化，即默认显示光滑曲线
    use_bspline = False

    while window.running:
        # 1. 交互事件处理
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append([pos[0], pos[1]])
            elif e.key == 'c':
                control_points.clear()
            elif e.key == 'b':
                use_bspline = not use_bspline
            elif e.key == 'a':
                show_pixelated = not show_pixelated # 切换模式

        # 2. 清理画布
        clear_pixels()
        
        # 3. 曲线绘制逻辑
        n_pts = len(control_points)
        if n_pts >= 2:
            # 根据状态选择曲线算法
            if use_bspline and n_pts >= 4:
                pts = compute_bspline_curve(control_points, NUM_SEGMENTS)
                current_color = BSPLINE_COLOR
            else:
                pts = compute_bezier_curve(control_points, NUM_SEGMENTS)
                current_color = BEZIER_COLOR
            
            # 将计算好的点传给 GPU
            valid_count = len(pts)
            curve_points_field.from_numpy(np.pad(pts, ((0, NUM_SEGMENTS + 1 - valid_count), (0, 0)), 'edge'))

            # --- 核心对比逻辑 ---
            if show_pixelated:
                # 模式 A：显示巨大像素块（极端走样）
                draw_pixelated_kernel(valid_count, current_color[0], current_color[1], current_color[2])
                canvas.set_image(pixels)
            else:
                # 模式 B：默认显示数学上的光滑曲线
                canvas.set_image(pixels)
                # 使用原生 lines 指令绘制，这是亚像素级、高精度、极致平滑的
                canvas.lines(curve_points_field, width=0.005, color=current_color)
        else:
            canvas.set_image(pixels)

        # 4. 绘制 UI 辅助元素（控制点与连线）
        gui_pts_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        if n_pts > 0:
            gui_pts_np[:n_pts] = np.array(control_points)
        gui_points.from_numpy(gui_pts_np)
        
        gui_idx_np = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
        if n_pts >= 2:
            for i in range(n_pts - 1):
                gui_idx_np[i * 2] = i
                gui_idx_np[i * 2 + 1] = i + 1
        gui_indices.from_numpy(gui_idx_np)
        
        # 渲染连线和节点
        if n_pts >= 2:
            canvas.lines(gui_points, width=0.002, indices=gui_indices, color=LINE_COLOR)
        canvas.circles(gui_points, radius=0.006, color=(0.75, 0.45, 0.45))
        
        window.show()

if __name__ == "__main__":
    main()