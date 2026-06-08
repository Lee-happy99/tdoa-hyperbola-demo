# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:45:40 2026

@author: ASUS
"""

# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Circle

st.set_page_config(page_title="TDOA Hyperbola Positioning", layout="wide")
st.title("📡 TDOA Localization: Time Difference → Hyperbolas → Intersection")
st.markdown("**Adjust station positions to see how two hyperbolas intersect at the target location**")

# ------------------- 侧边栏参数 -------------------
st.sidebar.header("📍 Station Coordinates (km)")

x1 = st.sidebar.slider("Station 1 - x", -20.0, 20.0, -8.0, 0.5)
y1 = st.sidebar.slider("Station 1 - y", -20.0, 20.0, 0.0, 0.5)

x2 = st.sidebar.slider("Station 2 - x", -20.0, 20.0, 8.0, 0.5)
y2 = st.sidebar.slider("Station 2 - y", -20.0, 20.0, 0.0, 0.5)

x3 = st.sidebar.slider("Station 3 - x", -20.0, 20.0, 0.0, 0.5)
y3 = st.sidebar.slider("Station 3 - y", -20.0, 20.0, 12.0, 0.5)

# 真实目标位置
true_x = st.sidebar.slider("🎯 True Target x", -20.0, 20.0, 5.0, 0.5)
true_y = st.sidebar.slider("🎯 True Target y", -20.0, 20.0, 5.0, 0.5)

# ------------------- 辅助函数 -------------------
def distance(p1, p2):
    return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

def tdoa_dist_diff(target, s1, s2):
    """返回目标到站2与到站1的距离差 (km)"""
    return distance(target, s2) - distance(target, s1)

def hyperbola_equation(p, s1, s2, delta_d):
    """双曲线方程残差: |d(p,s2) - d(p,s1) - delta_d|"""
    return abs(distance(p, s2) - distance(p, s1) - delta_d)

def find_hyperbola_intersection(s1, s2, s3, delta_d12, delta_d13, bounds=(-20,20)):
    """数值求解两条双曲线的交点（最小化残差）"""
    def objective(p):
        return hyperbola_equation(p, s1, s2, delta_d12) + hyperbola_equation(p, s1, s3, delta_d13)
    
    best_x, best_val = None, np.inf
    # 多起点搜索
    for x0 in np.linspace(bounds[0], bounds[1], 15):
        for y0 in np.linspace(bounds[0], bounds[1], 15):
            res = minimize(objective, [x0, y0], bounds=[bounds, bounds], method='L-BFGS-B')
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    return best_x

# ------------------- 计算距离差 -------------------
s1 = np.array([x1, y1])
s2 = np.array([x2, y2])
s3 = np.array([x3, y3])
target_true = np.array([true_x, true_y])

delta_d12 = tdoa_dist_diff(target_true, s1, s2)
delta_d13 = tdoa_dist_diff(target_true, s1, s3)

# ------------------- 绘制双曲线和交点 -------------------
fig, ax = plt.subplots(figsize=(9, 8))
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 1. 绘制三个侦察站
ax.plot(x1, y1, 'o', color='blue', markersize=10, label='Station 1')
ax.plot(x2, y2, 'o', color='blue', markersize=10, label='Station 2')
ax.plot(x3, y3, 'o', color='blue', markersize=10, label='Station 3')
# 添加文字标注
ax.text(x1, y1-1.2, "S1", color='blue', fontsize=10, ha='center', weight='bold')
ax.text(x2, y2-1.2, "S2", color='blue', fontsize=10, ha='center', weight='bold')
ax.text(x3, y3-1.2, "S3", color='blue', fontsize=10, ha='center', weight='bold')

# 2. 真实目标
ax.plot(true_x, true_y, 'r*', markersize=18, label='True Target')
ax.text(true_x, true_y+1.2, "Target", color='red', fontsize=10, ha='center', weight='bold')

# 3. 绘制两条双曲线（通过网格等值线）
grid_res = 100
x_grid = np.linspace(-20, 20, grid_res)
y_grid = np.linspace(-20, 20, grid_res)
X, Y = np.meshgrid(x_grid, y_grid)

# 计算到站1和站2的距离差等值面
dist_diff12 = np.hypot(X - s2[0], Y - s2[1]) - np.hypot(X - s1[0], Y - s1[1])
dist_diff13 = np.hypot(X - s3[0], Y - s3[1]) - np.hypot(X - s1[0], Y - s1[1])

# 绘制距离差等于给定值的等高线（双曲线）
contour12 = ax.contour(X, Y, dist_diff12, levels=[delta_d12], colors='red', linewidths=2, linestyles='-')
contour13 = ax.contour(X, Y, dist_diff13, levels=[delta_d13], colors='green', linewidths=2, linestyles='-')

# 手动创建图例（用自定义线条）
ax.plot([], [], color='red', linewidth=2, label=f'Hyperbola (S1-S2) Δd = {delta_d12:.2f} km')
ax.plot([], [], color='green', linewidth=2, label=f'Hyperbola (S1-S3) Δd = {delta_d13:.2f} km')

# 4. 求双曲线交点（估计位置）
est_pos = find_hyperbola_intersection(s1, s2, s3, delta_d12, delta_d13)
if est_pos is not None:
    ax.plot(est_pos[0], est_pos[1], 'o', color='orange', markersize=10, label='Estimated Position')
    ax.text(est_pos[0], est_pos[1]+1.2, "Estimate", color='orange', fontsize=10, ha='center')
    error = np.hypot(est_pos[0]-true_x, est_pos[1]-true_y)
    st.sidebar.metric("📍 Localization Error", f"{error:.2f} km")
else:
    st.sidebar.warning("⚠️ Intersection not found, adjust station geometry")

ax.legend(loc='upper right')

st.pyplot(fig)

# 教学说明
st.info("""
**📖 Principle of TDOA Hyperbola Positioning**

1. **Time Difference of Arrival (TDOA)** is converted to **distance difference**:  
   `Δd = c · Δt`. Here we use `c=1 km/μs` for simplicity.

2. **Hyperbola definition**: All points with a constant distance difference `Δd` to two fixed stations form a hyperbola.

3. **Two hyperbolas** (from station pairs S1-S2 and S1-S3) intersect at the target location.

4. **Drag station positions** in the sidebar to see hyperbolas change in real time. The estimated position (orange dot) should approach the true target (red star).

> 💡 **Try this**: Move station 3 to be collinear with S1 and S2 → hyperbolas become nearly parallel and intersection becomes unstable → large localization error.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📱 Student Access")
app_url = "https://你的应用名.streamlit.app"  # 部署后替换为实际链接
st.sidebar.info(f"Copy this link to generate QR code for students:\n\n`{app_url}`")