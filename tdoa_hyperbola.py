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
import matplotlib.font_manager as fm
import os

# ------------------- 中文字体配置 -------------------
font_path = os.path.join(os.path.dirname(__file__), 'simhei.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    st.warning("字体文件 simhei.ttf 未找到，将使用默认字体（可能显示方块）")
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 页面配置 -------------------
st.set_page_config(page_title="TDOA双曲线定位", layout="wide")

# 修正标题被遮住的问题：增加上边距和行高
st.markdown("""
<style>
    h1 {
        font-size: 1.8rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.2rem !important;
        line-height: 1.2 !important;
    }
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📡 TDOA时差定位：双曲线相交锁定目标")
st.markdown("**拖动左侧滑块调整三个侦察站的位置，观察双曲线如何实时变化并交于目标点**")

# ------------------- 侧边栏（紧凑布局） -------------------
st.sidebar.header("📍 侦察站坐标 (km)")

col1, col2 = st.sidebar.columns(2)
with col1:
    x1 = st.slider("站1 x", -20.0, 20.0, -8.0, 0.5)
with col2:
    y1 = st.slider("站1 y", -20.0, 20.0, 0.0, 0.5)

col1, col2 = st.sidebar.columns(2)
with col1:
    x2 = st.slider("站2 x", -20.0, 20.0, 8.0, 0.5)
with col2:
    y2 = st.slider("站2 y", -20.0, 20.0, 0.0, 0.5)

col1, col2 = st.sidebar.columns(2)
with col1:
    x3 = st.slider("站3 x", -20.0, 20.0, 0.0, 0.5)
with col2:
    y3 = st.slider("站3 y", -20.0, 20.0, 12.0, 0.5)

st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
with col1:
    true_x = st.slider("🎯 真实目标 x", -20.0, 20.0, 5.0, 0.5)
with col2:
    true_y = st.slider("🎯 真实目标 y", -20.0, 20.0, 5.0, 0.5)

# ------------------- 辅助函数 -------------------
def distance(p1, p2):
    return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

def tdoa_dist_diff(target, s1, s2):
    return distance(target, s2) - distance(target, s1)

def hyperbola_equation(p, s1, s2, delta_d):
    return abs(distance(p, s2) - distance(p, s1) - delta_d)

def find_hyperbola_intersection(s1, s2, s3, delta_d12, delta_d13, bounds=(-20,20)):
    def objective(p):
        return hyperbola_equation(p, s1, s2, delta_d12) + hyperbola_equation(p, s1, s3, delta_d13)
    best_x, best_val = None, np.inf
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

# ------------------- 绘图 -------------------
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 侦察站
ax.plot(x1, y1, 'o', color='blue', markersize=10)
ax.plot(x2, y2, 'o', color='blue', markersize=10)
ax.plot(x3, y3, 'o', color='blue', markersize=10)
ax.text(x1, y1-1.2, "侦察站1", color='blue', fontsize=10, ha='center', weight='bold')
ax.text(x2, y2-1.2, "侦察站2", color='blue', fontsize=10, ha='center', weight='bold')
ax.text(x3, y3-1.2, "侦察站3", color='blue', fontsize=10, ha='center', weight='bold')

# 真实目标
ax.plot(true_x, true_y, 'r*', markersize=18, label='真实目标')
ax.text(true_x + 1.0, true_y + 1.2, "目标点", color='red', fontsize=10, ha='left', weight='bold')

# 绘制双曲线
grid_res = 100
x_grid = np.linspace(-20, 20, grid_res)
y_grid = np.linspace(-20, 20, grid_res)
X, Y = np.meshgrid(x_grid, y_grid)

dist_diff12 = np.hypot(X - s2[0], Y - s2[1]) - np.hypot(X - s1[0], Y - s1[1])
dist_diff13 = np.hypot(X - s3[0], Y - s3[1]) - np.hypot(X - s1[0], Y - s1[1])

ax.contour(X, Y, dist_diff12, levels=[delta_d12], colors='red', linewidths=2, linestyles='-')
ax.contour(X, Y, dist_diff13, levels=[delta_d13], colors='green', linewidths=2, linestyles='-')

# 图例
ax.plot([], [], color='red', linewidth=2, label=f'双曲线 (站1-站2) 距离差 = {delta_d12:.2f} km')
ax.plot([], [], color='green', linewidth=2, label=f'双曲线 (站1-站3) 距离差 = {delta_d13:.2f} km')

# 估计点（紫色）
est_pos = find_hyperbola_intersection(s1, s2, s3, delta_d12, delta_d13)
if est_pos is not None:
    ax.plot(est_pos[0], est_pos[1], 'o', color='purple', markersize=10, label='TDOA估计位置')
    ax.text(est_pos[0] - 1.5, est_pos[1] - 1.2, "估计点", color='purple', fontsize=10, ha='center', weight='bold')
    error = np.hypot(est_pos[0]-true_x, est_pos[1]-true_y)
    st.sidebar.metric("📍 定位误差", f"{error:.2f} km")
else:
    st.sidebar.warning("⚠️ 未找到交点，请调整站址避免三站共线")

ax.legend(loc='upper right')
st.pyplot(fig, use_container_width=True)

# 教学说明（已删除几何稀释效应）
with st.expander("📖 TDOA双曲线定位原理（点击展开）"):
    st.markdown("""
    - **到达时间差（TDOA）** → **距离差**：`Δd = c · Δt`（本演示中设 `c=1 km/μs` 简化）。  
    - **双曲线定义**：到两个固定点距离差为恒定值的所有点构成一条双曲线。  
    - **两条双曲线**（来自站1-站2 和 站1-站3）的交点就是目标位置。  
    - **操作提示**：拖动左侧滑块调整侦察站或目标位置，观察双曲线实时变化，估计点（紫色圆）应逼近真实目标（红色五角星）。
    """)

# 注释掉侧边栏的访问链接部分
# st.sidebar.markdown("---")
# st.sidebar.subheader("📱 学生扫码访问")
# app_url = "https://你的应用名.streamlit.app"
# st.sidebar.info(f"复制此链接生成二维码：\n\n`{app_url}`")