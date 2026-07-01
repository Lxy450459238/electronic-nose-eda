# 全阶段时序示意图 (Figure 5.3)
# 展示电子鼻单次实验完整六阶段: P13→P14→P15→P16→P17→P18
# 含进样阶段探测逻辑(P15优先/P17回退) 小图
# 输出: full_stage_timeseries.png (300 DPI, 中文标注)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import os

# ===== 字体配置 =====
font_path = 'C:/Windows/Fonts/simhei.ttf'
if not os.path.exists(font_path):
    font_path = 'C:/Windows/Fonts/msyh.ttf'
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 生成模拟全阶段数据 =====
np.random.seed(42)
noise = 2.5
baseline_val = 500.0

# 各阶段长度（采样点数）
stage_lengths = {
    'P13': 100,
    'P14': 80,
    'P15': 200,
    'P16': 160,
    'P17': 180,
    'P18': 130,
}

# 阶段配色（与新代码阶段背景色一致）
stage_colors = {
    'P13': '#e8f4f8',   # 浅蓝
    'P14': '#fff4e6',   # 浅橙
    'P15': '#f0f8e8',   # 浅绿
    'P16': '#fce4ec',   # 浅粉
    'P17': '#e8f0fe',   # 浅紫蓝
    'P18': '#f3e5f5',   # 浅紫
}

stage_edge_colors = {
    'P13': '#78909C',
    'P14': '#FFB74D',
    'P15': '#66BB6A',
    'P16': '#EF5350',
    'P17': '#5C6BC0',
    'P18': '#AB47BC',
}

stage_labels_cn = {
    'P13': 'P13\n基线',
    'P14': 'P14\n二次基线',
    'P15': 'P15\n主进样',
    'P16': 'P16\n主清洗',
    'P17': 'P17\n二次进样',
    'P18': 'P18\n最终清洗',
}

# 构建完整时间序列
all_signal = []
all_t = []
stage_boundaries = [0]  # 各阶段起始索引
current_t = 0

# P13: 基线
p13_signal = baseline_val + np.random.randn(stage_lengths['P13']) * noise
all_signal.append(p13_signal)
all_t.append(np.arange(stage_lengths['P13']))
current_t += stage_lengths['P13']
stage_boundaries.append(current_t)

# P14: 二次基线（与P13连续，微小的系统漂移）
p14_base = baseline_val + 8
p14_signal = p14_base + np.random.randn(stage_lengths['P14']) * noise
all_signal.append(p14_signal)
all_t.append(np.arange(stage_lengths['P14']) + current_t)
current_t += stage_lengths['P14']
stage_boundaries.append(current_t)

# P15: 主进样（S型爬坡 → 平台）
p15_pts = stage_lengths['P15']
ramp_len_15 = 70
platform_len_15 = p15_pts - ramp_len_15
p15_peak = 650.0

# 爬坡段
x_ramp = np.linspace(0, np.pi, ramp_len_15)
ramp_curve = (p14_base + 5) + (p15_peak - p14_base - 5) * (1 - np.cos(x_ramp)) / 2
ramp_noisy = ramp_curve + np.random.randn(ramp_len_15) * noise * 1.5
# 平台段
platform = p15_peak + np.random.randn(platform_len_15) * noise
p15_signal = np.concatenate([ramp_noisy, platform])
all_signal.append(p15_signal)
all_t.append(np.arange(p15_pts) + current_t)
current_t += p15_pts
stage_boundaries.append(current_t)

# P16: 主清洗（指数衰减 → 基线）
p16_pts = stage_lengths['P16']
decay_tau = 45
decay = p15_peak + (p14_base - p15_peak) * (1 - np.exp(-np.arange(p16_pts) / decay_tau))
# 添加少量过冲（洗过头）
overshoot = -8 * np.exp(-np.arange(p16_pts) / 25) * (np.arange(p16_pts) < 60)
p16_signal = decay + overshoot + np.random.randn(p16_pts) * noise * 1.2
all_signal.append(p16_signal)
all_t.append(np.arange(p16_pts) + current_t)
current_t += p16_pts
stage_boundaries.append(current_t)

# P17: 二次进样（类似P15但峰值略低）
p17_pts = stage_lengths['P17']
ramp_len_17 = 65
platform_len_17 = p17_pts - ramp_len_17
p17_peak = 630.0  # 略低于P15，反映传感器疲劳

x_ramp2 = np.linspace(0, np.pi, ramp_len_17)
ramp_curve2 = (p14_base - 2) + (p17_peak - p14_base + 2) * (1 - np.cos(x_ramp2)) / 2
ramp_noisy2 = ramp_curve2 + np.random.randn(ramp_len_17) * noise * 1.5
platform2 = p17_peak + np.random.randn(platform_len_17) * noise
p17_signal = np.concatenate([ramp_noisy2, platform2])
all_signal.append(p17_signal)
all_t.append(np.arange(p17_pts) + current_t)
current_t += p17_pts
stage_boundaries.append(current_t)

# P18: 最终清洗
p18_pts = stage_lengths['P18']
decay2 = p17_peak + (p14_base - 5 - p17_peak) * (1 - np.exp(-np.arange(p18_pts) / 40))
p18_signal = decay2 + np.random.randn(p18_pts) * noise * 1.2
all_signal.append(p18_signal)
all_t.append(np.arange(p18_pts) + current_t)
current_t += p18_pts
stage_boundaries.append(current_t)

# 合并
full_signal = np.concatenate(all_signal)
full_t = np.concatenate(all_t)
total_pts = len(full_signal)

# ===== 2. 绘图 =====
fig = plt.figure(figsize=(20, 10))

# ---- 2.1 主图：全阶段时序 ----
ax_main = fig.add_axes([0.06, 0.18, 0.90, 0.78])  # [left, bottom, width, height]

# 阶段背景色
stage_names = ['P13', 'P14', 'P15', 'P16', 'P17', 'P18']
for i, sn in enumerate(stage_names):
    x0 = stage_boundaries[i]
    x1 = stage_boundaries[i + 1]
    ax_main.axvspan(x0, x1, alpha=0.45, facecolor=stage_colors[sn],
                    edgecolor=stage_edge_colors[sn], linewidth=1.8, linestyle='-', zorder=1)
    # 阶段标签（顶部）
    ax_main.text((x0 + x1) / 2, full_signal.max() + 28, stage_labels_cn[sn],
                 fontproperties=prop, fontsize=12, fontweight='bold',
                 color=stage_edge_colors[sn], ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                           edgecolor=stage_edge_colors[sn], alpha=0.85))

# 绘制响应曲线
ax_main.plot(full_t, full_signal, color='#37474F', linewidth=1.3, zorder=3, alpha=0.9)

# 阶段分界线（垂直虚线）
for i, sn in enumerate(stage_names):
    x0 = stage_boundaries[i]
    ax_main.axvline(x=x0, color=stage_edge_colors[sn], linewidth=1.5,
                    linestyle='--', alpha=0.5, zorder=2)

# 最后一条分界线
ax_main.axvline(x=stage_boundaries[-1], color='#78909C', linewidth=1.2,
                linestyle='--', alpha=0.4, zorder=2)

# ---- 2.2 进样阶段高亮标注 ----
# P15 主进样 → 绿色边框加粗
x_p15_start = stage_boundaries[2]
x_p15_end = stage_boundaries[3]
ax_main.add_patch(plt.Rectangle(
    (x_p15_start, full_signal.min() - 20), x_p15_end - x_p15_start,
    full_signal.max() - full_signal.min() + 45,
    fill=False, edgecolor='#2E7D32', linewidth=3.5, linestyle='-',
    zorder=5, alpha=0.85
))
ax_main.annotate('核心进样阶段\nsample_stage = P15',
                 xy=(x_p15_start + (x_p15_end - x_p15_start) * 0.55, full_signal.max()),
                 xytext=(x_p15_start + (x_p15_end - x_p15_start) * 0.75, full_signal.max() + 65),
                 fontproperties=prop, fontsize=11, fontweight='bold',
                 color='#1B5E20', ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                           edgecolor='#2E7D32', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.5))

# P17 二次进样 → 蓝紫边框（备选标注）
x_p17_start = stage_boundaries[4]
x_p17_end = stage_boundaries[5]
ax_main.add_patch(plt.Rectangle(
    (x_p17_start, full_signal.min() - 20), x_p17_end - x_p17_start,
    full_signal.max() - full_signal.min() + 45,
    fill=False, edgecolor='#5C6BC0', linewidth=3.5, linestyle='--',
    zorder=5, alpha=0.70
))
ax_main.annotate('备选进样阶段\nP17 (P15缺失时启用)',
                 xy=(x_p17_start + (x_p17_end - x_p17_start) * 0.45, full_signal.max()),
                 xytext=(x_p17_start + (x_p17_end - x_p17_start) * 0.70, full_signal.max() + 65),
                 fontproperties=prop, fontsize=10, fontweight='bold',
                 color='#283593', ha='center',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8EAF6',
                           edgecolor='#5C6BC0', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', color='#5C6BC0', lw=2.2))

# ---- 2.3 坐标轴 ----
ax_main.set_xlabel('采样时刻 (Sampling Time Points)', fontproperties=prop, fontsize=14)
ax_main.set_ylabel('传感器响应值 (Sensor Response)', fontproperties=prop, fontsize=14)
ax_main.set_title('图5.3  电子鼻单次实验全阶段响应时序示意图',
                  fontproperties=prop, fontsize=16, fontweight='bold', pad=10)
ax_main.set_xlim(-5, total_pts + 5)
ax_main.set_ylim(full_signal.min() - 35, full_signal.max() + 85)
ax_main.grid(True, alpha=0.15, linestyle='--')
ax_main.tick_params(labelsize=10)

# ---- 2.4 图例（底部居中）----
legend_elements = [
    mpatches.Patch(facecolor=stage_colors['P13'], edgecolor=stage_edge_colors['P13'],
                   label='P13 基线', alpha=0.7),
    mpatches.Patch(facecolor=stage_colors['P14'], edgecolor=stage_edge_colors['P14'],
                   label='P14 二次基线', alpha=0.7),
    mpatches.Patch(facecolor=stage_colors['P15'], edgecolor=stage_edge_colors['P15'],
                   label='P15 主进样 (sample_stage)', alpha=0.7),
    mpatches.Patch(facecolor=stage_colors['P16'], edgecolor=stage_edge_colors['P16'],
                   label='P16 主清洗', alpha=0.7),
    mpatches.Patch(facecolor=stage_colors['P17'], edgecolor=stage_edge_colors['P17'],
                   label='P17 二次进样 (备选)', alpha=0.7),
    mpatches.Patch(facecolor=stage_colors['P18'], edgecolor=stage_edge_colors['P18'],
                   label='P18 最终清洗', alpha=0.7),
]
legend = ax_main.legend(handles=legend_elements, loc='lower center',
                        fontsize=10.5, framealpha=0.9, edgecolor='#BDBDBD',
                        ncol=6, title='阶段图例',
                        bbox_to_anchor=(0.5, -0.17))
legend.get_title().set_fontproperties(prop)
legend.get_title().set_fontsize(12)
legend.get_title().set_fontweight('bold')

# ---- 2.5 底部说明文字 ----
fig.text(0.06, 0.03,
         '注：P16为两次进样(P15与P17)之间的过渡清洗阶段。算法按 P15→P17 优先级探测进样阶段。',
         fontproperties=prop, fontsize=10, color='#616161', fontstyle='italic')

# ===== 3. 保存 =====
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'full_stage_timeseries.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"[OK] 全阶段时序示意图已保存至:\n  {output_path}")
print(f"  尺寸: {fig.get_size_inches()[0]:.1f}×{fig.get_size_inches()[1]:.1f} inch")
print(f"  DPI: 300")
