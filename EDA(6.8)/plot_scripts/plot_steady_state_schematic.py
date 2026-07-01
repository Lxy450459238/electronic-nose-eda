# 稳态检测原理示意图
# 展示滑动窗口 + 前瞻验证的稳态搜索算法流程
# 输出: steady_state_schematic.png (300 DPI, 中文标注)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
import os

# ===== 字体配置 =====
font_path = 'C:/Windows/Fonts/simhei.ttf'
if not os.path.exists(font_path):
    font_path = 'C:/Windows/Fonts/msyh.ttf'  # 备选微软雅黑
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 生成模拟传感器响应数据 =====
np.random.seed(42)
n_total = 220          # 总采样点数
t = np.arange(n_total)

# 阶段模拟：基线 → 响应爬坡 → 稳态平台
# 0-29: 基线阶段 (≈500)
# 30-99: 响应爬坡 (500 → 650)
# 100-219: 稳态平台 (≈650, 小幅波动)

baseline_val = 500.0
steady_val = 650.0
noise_level = 3.0      # 稳态小噪声

signal = np.zeros(n_total)

# 基线段 (0-29)
signal[0:30] = baseline_val + np.random.randn(30) * noise_level

# 爬坡段 (30-99): S型上升曲线
climb_len = 70
x_climb = np.linspace(0, np.pi, climb_len)
climb_curve = baseline_val + (steady_val - baseline_val) * (1 - np.cos(x_climb)) / 2
signal[30:100] = climb_curve + np.random.randn(climb_len) * noise_level * 1.5

# 稳态平台段 (100-219): 小幅波动
platform_len = 120
signal[100:] = steady_val + np.random.randn(platform_len) * noise_level

# ===== 2. 模拟算法找到的稳态窗口位置 =====
window_size = 10
verify_len = 20

# 算法找到第一个稳态窗口的位置（假设在 i=110 处）
found_idx = 110
window_start = found_idx
window_end = found_idx + window_size
verify_start = window_end
verify_end = window_end + verify_len
stable_start = window_start
stable_end = verify_end

# 稳态均值
stable_segment = signal[stable_start:stable_end]
steady_mean = np.mean(stable_segment)

# ===== 3. 绘图 =====
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# ---- 3.1 背景阶段色块 ----
# 爬坡段背景
ax.axvspan(30, 100, alpha=0.08, color='#E8845C', zorder=0)
# 稳态平台背景
ax.axvspan(100, n_total - 1, alpha=0.05, color='#4CAF50', zorder=0)

# ---- 3.2 绘制响应曲线 ----
ax.plot(t[0:30], signal[0:30], color='#90A4AE', linewidth=1.8, alpha=0.7, label='基线段')
ax.plot(t[30:100], signal[30:100], color='#E8845C', linewidth=1.8, alpha=0.85, label='响应爬坡段')
ax.plot(t[100:], signal[100:], color='#2E7D32', linewidth=1.8, alpha=0.85, label='稳态平台段')

# ---- 3.3 滑动搜索窗口（高亮矩形框）----
ax.axvspan(window_start, window_end, alpha=0.25, facecolor='#1976D2', zorder=2,
           edgecolor='#0D47A1', linewidth=2.5, linestyle='-')
# 窗口下方标注
ax.annotate('', xy=(window_start, signal.min() - 25), xytext=(window_end, signal.min() - 25),
            arrowprops=dict(arrowstyle='<->', color='#0D47A1', lw=3))

# ---- 3.4 验证段（高亮矩形框）----
ax.axvspan(verify_start, verify_end, alpha=0.20, facecolor='#FF6F00', zorder=2,
           edgecolor='#BF360C', linewidth=2.5, linestyle='--')

# ---- 3.5 稳态取值虚线 ----
ax.axhline(y=steady_mean, color='#C62828', linewidth=2.5, linestyle='--',
           alpha=0.85, zorder=3)
ax.axvline(x=stable_end, ymin=0.05, ymax=0.78, color='#C62828',
           linewidth=1.5, linestyle=':', alpha=0.5, zorder=2)

# ---- 3.6 标注文字 ----
# 爬坡段标注
ax.annotate('响应爬坡段\n(Response Ramp)',
            xy=(65, signal[65]),
            xytext=(65, signal.max() + 30),
            fontproperties=prop, fontsize=13, fontweight='bold',
            color='#BF360C', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0',
                      edgecolor='#E8845C', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#BF360C', lw=2.0))

# 滑动窗口标注
ax.annotate('滑动搜索窗口\n(Window={}pts)'.format(window_size),
            xy=((window_start + window_end) / 2, signal[window_start:window_end].max()),
            xytext=((window_start + window_end) / 2, signal.max() + 55),
            fontproperties=prop, fontsize=13, fontweight='bold',
            color='#0D47A1', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD',
                      edgecolor='#1976D2', alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=2.0))

# 验证段标注
ax.annotate('前瞻验证段\n(Verification={}pts)'.format(verify_len),
            xy=(verify_start + verify_len / 2, signal[verify_start:verify_end].min()),
            xytext=(verify_start + verify_len / 2, signal.min() - 55),
            fontproperties=prop, fontsize=13, fontweight='bold',
            color='#BF360C', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E1',
                      edgecolor='#FF6F00', alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#BF360C', lw=2.0))

# 稳态取值点标注
ax.annotate('稳态取值\nμ={:.1f}'.format(steady_mean),
            xy=(stable_end + 5, steady_mean),
            xytext=(stable_end + 30, steady_mean + 25),
            fontproperties=prop, fontsize=13, fontweight='bold',
            color='#C62828', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE',
                      edgecolor='#C62828', alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.0,
                            connectionstyle='arc3,rad=0.3'))

# 搜索方向箭头（底部大箭头）
ax.annotate('', xy=(found_idx - 15, signal.min() - 75),
            xytext=(30, signal.min() - 75),
            arrowprops=dict(arrowstyle='->', color='#546E7A', lw=3.5,
                            connectionstyle='arc3,rad=-0.1'))
ax.text(70, signal.min() - 90, '滑动搜索方向 →',
        fontproperties=prop, fontsize=12, color='#546E7A',
        ha='center', fontstyle='italic')

# ---- 3.7 失败路径示意（旁注）----
ax.annotate('未通过→\n窗口右移',
            xy=(60, signal[60]),
            xytext=(15, signal.max() + 55),
            fontproperties=prop, fontsize=9, color='#78909C',
            ha='center', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECEFF1',
                      edgecolor='#B0BEC5', alpha=0.8))

# ---- 3.9 坐标轴设置 ----
ax.set_xlabel('采样时刻 (Sampling Time Points)', fontproperties=prop, fontsize=14)
ax.set_ylabel('传感器响应值 (Sensor Response)', fontproperties=prop, fontsize=14)
ax.set_title('稳态检测原理示意图 — 滑动窗口 + 前瞻验证算法',
             fontproperties=prop, fontsize=17, fontweight='bold', pad=18)

ax.set_xlim(-2, n_total + 5)
ax.set_ylim(signal.min() - 100, signal.max() + 75)
ax.grid(True, alpha=0.2, linestyle='--')

# ---- 3.10 图例 ----
legend_elements = [
    plt.Line2D([0], [0], color='#E8845C', lw=2.5, label='响应爬坡段'),
    plt.Line2D([0], [0], color='#2E7D32', lw=2.5, label='稳态平台段'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#1976D2', alpha=0.25,
                  edgecolor='#0D47A1', linewidth=2, label='滑动搜索窗口 (Window)'),
    plt.Rectangle((0, 0), 1, 1, facecolor='#FF6F00', alpha=0.20,
                  edgecolor='#BF360C', linewidth=2, linestyle='--', label='前瞻验证段 (Verification)'),
    plt.Line2D([0], [0], color='#C62828', lw=2.5, linestyle='--', label='稳态取值 (μ)'),
]
ax.legend(handles=legend_elements, loc='lower right', prop=prop, fontsize=11,
          framealpha=0.9, edgecolor='#BDBDBD')

plt.tight_layout()

# ===== 4. 保存 =====
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'steady_state_schematic.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"[OK] 稳态检测原理示意图已保存至:\n  {output_path}")
print(f"  尺寸: {fig.get_size_inches()[0]:.1f}×{fig.get_size_inches()[1]:.1f} inch")
print(f"  DPI: 300")
