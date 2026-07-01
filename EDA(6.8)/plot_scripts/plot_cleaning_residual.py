# 清洗过程与残留率定义示意图 (Figure 6.1)
# 展示基线、响应峰值、清洗终点三个关键位置及残留率物理定义
# 输出: cleaning_residual_rate.png (300 DPI, 中文标注)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ===== 字体配置 =====
font_path = 'C:/Windows/Fonts/simhei.ttf'
if not os.path.exists(font_path):
    font_path = 'C:/Windows/Fonts/msyh.ttf'
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 生成模拟数据 =====
np.random.seed(123)
noise = 1.8

# 阶段划分（采样点）
n_baseline = 80       # 基线段
n_ramp = 50           # 响应爬坡
n_plateau = 60        # 稳态平台
n_decay = 100         # 清洗衰减
n_post = 50           # 清洗后延续

total_n = n_baseline + n_ramp + n_plateau + n_decay + n_post

# 基准值
baseline_val = 500.0
peak_val = 680.0

# 基线段
t_baseline = np.arange(n_baseline)
sig_baseline = baseline_val + np.random.randn(n_baseline) * noise

# 爬坡段（S型）
t_ramp = np.arange(n_ramp) + n_baseline
x_ramp = np.linspace(0, np.pi, n_ramp)
ramp_curve = baseline_val + (peak_val - baseline_val) * (1 - np.cos(x_ramp)) / 2
sig_ramp = ramp_curve + np.random.randn(n_ramp) * noise * 1.3

# 平台段（稳态进样）
t_plateau = np.arange(n_plateau) + n_baseline + n_ramp
sig_plateau = peak_val + np.random.randn(n_plateau) * noise

# 清洗衰减段（指数衰减，残留一部分未回到基线）
t_decay = np.arange(n_decay) + n_baseline + n_ramp + n_plateau

# 清洗末端值：故意残留 ~15%，展示残留率概念
clean_end_val = baseline_val + (peak_val - baseline_val) * 0.15  # 残留15%
tau = 38
clean_curve = peak_val + (clean_end_val - peak_val) * (1 - np.exp(-np.arange(n_decay) / tau))
sig_decay = clean_curve + np.random.randn(n_decay) * noise * 1.2

# 清洗后延续
t_post = np.arange(n_post) + n_baseline + n_ramp + n_plateau + n_decay
sig_post = clean_end_val + np.random.randn(n_post) * noise * 0.6 + \
           np.linspace(0, -3, n_post)  # 微弱继续漂移

# 合并
t_all = np.arange(total_n)
signal = np.concatenate([sig_baseline, sig_ramp, sig_plateau, sig_decay, sig_post])

# ===== 2. 计算关键值 =====
mu_baseline = np.mean(sig_baseline[-30:])         # 基线稳态均值
mu_sample = np.mean(sig_plateau[-30:])             # 进样稳态均值
mu_clean = np.mean(sig_post[-30:])                 # 清洗末端稳态均值

response_amplitude = mu_sample - mu_baseline       # 总响应幅度
residual = mu_clean - mu_baseline                   # 残留幅度
residual_rate = residual / response_amplitude       # 残留率

# ===== 3. 绘图 =====
fig, ax = plt.subplots(1, 1, figsize=(16, 7.5))

# ---- 3.1 阶段背景色 ----
baseline_end = n_baseline
ramp_end = n_baseline + n_ramp
plateau_end = n_baseline + n_ramp + n_plateau
decay_end = n_baseline + n_ramp + n_plateau + n_decay

# 基线段背景
ax.axvspan(0, baseline_end, alpha=0.12, facecolor='#BBDEFB',
           edgecolor='#64B5F6', linewidth=1.0, linestyle='-')
# 进样段背景（爬坡+平台）
ax.axvspan(baseline_end, plateau_end, alpha=0.12, facecolor='#C8E6C9',
           edgecolor='#81C784', linewidth=1.0, linestyle='-')
# 清洗段背景（衰减+延续）
ax.axvspan(plateau_end, total_n, alpha=0.12, facecolor='#FFCDD2',
           edgecolor='#E57373', linewidth=1.0, linestyle='-')

# ---- 3.2 绘制响应曲线 ----
ax.plot(t_all, signal, color='#37474F', linewidth=1.5, zorder=3, alpha=0.92)

# ---- 3.3 三条关键水平线 ----
# 基线均值线（蓝色虚线）
ax.axhline(y=mu_baseline, color='#1565C0', linewidth=2.2, linestyle='--',
           alpha=0.80, zorder=4, xmin=0.0, xmax=0.97)

# 进样稳态均值线（绿色虚线）
ax.axhline(y=mu_sample, color='#2E7D32', linewidth=2.2, linestyle='--',
           alpha=0.80, zorder=4, xmin=0.35, xmax=0.68)

# 清洗末端均值线（红色虚线）
ax.axhline(y=mu_clean, color='#C62828', linewidth=2.2, linestyle='--',
           alpha=0.80, zorder=4, xmin=0.70, xmax=0.97)

# ---- 3.4 垂直标注线（双箭头：响应幅度 vs 残留幅度）----

# 响应幅度：基线 → 峰值（绿色双箭头）
y_mid_resp = (mu_baseline + mu_sample) / 2
ax.annotate('', xy=(t_plateau[len(t_plateau)//2], mu_baseline),
            xytext=(t_plateau[len(t_plateau)//2], mu_sample),
            arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=3.5))
# 响应幅度标注
ax.text(t_plateau[len(t_plateau)//2] + 12, y_mid_resp,
        '响应幅度\n|μ_sample - μ_baseline|',
        fontproperties=prop, fontsize=11, fontweight='bold',
        color='#1B5E20', ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                  edgecolor='#43A047', alpha=0.90))

# 残留幅度：基线 → 清洗末端（红色双箭头）
y_mid_resid = (mu_baseline + mu_clean) / 2
ax.annotate('', xy=(t_post[len(t_post)//2], mu_baseline),
            xytext=(t_post[len(t_post)//2], mu_clean),
            arrowprops=dict(arrowstyle='<->', color='#C62828', lw=3.5))
# 残留幅度标注
ax.text(t_post[len(t_post)//2] + 12, y_mid_resid,
        '残留幅度\n|μ_clean - μ_baseline|',
        fontproperties=prop, fontsize=11, fontweight='bold',
        color='#B71C1C', ha='left', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE',
                  edgecolor='#E53935', alpha=0.90))

# ---- 3.5 三个关键位置标注 ----
# 基线
ax.annotate('基线 μ_baseline',
            xy=(n_baseline - 20, mu_baseline),
            xytext=(n_baseline - 60, mu_baseline - 55),
            fontproperties=prop, fontsize=11.5, fontweight='bold',
            color='#0D47A1', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD',
                      edgecolor='#1565C0', alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2.2,
                            connectionstyle='arc3,rad=0.2'))

# 响应峰值
ax.annotate('响应峰值 μ_sample',
            xy=(t_plateau[len(t_plateau)//2], mu_sample),
            xytext=(t_plateau[len(t_plateau)//2] - 10, mu_sample + 42),
            fontproperties=prop, fontsize=11.5, fontweight='bold',
            color='#1B5E20', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                      edgecolor='#2E7D32', alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2.2))

# 清洗终点
ax.annotate('清洗终点 μ_clean',
            xy=(t_post[len(t_post)//2], mu_clean),
            xytext=(t_post[len(t_post)//2] + 40, mu_clean + 40),
            fontproperties=prop, fontsize=11.5, fontweight='bold',
            color='#B71C1C', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE',
                      edgecolor='#C62828', alpha=0.92),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.2,
                            connectionstyle='arc3,rad=-0.3'))

# ---- 3.6 阶段标签 ----
stage_annos = [
    (n_baseline / 2, '基线阶段\nP13', '#1565C0'),
    (baseline_end + (n_ramp + n_plateau) / 2, '进样阶段\nP15', '#2E7D32'),
    (plateau_end + (n_decay + n_post) / 2, '清洗阶段\nP16', '#C62828'),
]
for x_pos, label, color in stage_annos:
    ax.text(x_pos, signal.min() - 52, label,
            fontproperties=prop, fontsize=12, fontweight='bold',
            color=color, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor=color, alpha=0.88))

# ---- 3.7 坐标轴 ----
ax.set_xlabel('采样时刻 (Sampling Time Points)', fontproperties=prop, fontsize=14)
ax.set_ylabel('传感器响应值 (Sensor Response)', fontproperties=prop, fontsize=14)
ax.set_title('图6.1  传感器响应—清洗过程与残留率定义示意图',
             fontproperties=prop, fontsize=16, fontweight='bold', pad=15)
ax.set_xlim(-5, total_n + 5)
ax.set_ylim(signal.min() - 70, signal.max() + 55)
ax.grid(True, alpha=0.13, linestyle='--')
ax.tick_params(labelsize=10)

# ---- 3.8 底部说明 ----
fig.text(0.06, 0.02,
         '注：残留率(Recovery Rate)衡量清洗后传感器偏离基线的程度。'
         'RR越大表示清洗越不彻底，残留越多；RR为负表示越过基线（过度清洗或漂移）。',
         fontproperties=prop, fontsize=10, color='#616161', fontstyle='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])

# ===== 4. 保存 =====
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'cleaning_residual_rate.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"[OK] 清洗残留率示意图已保存至:\n  {output_path}")
print(f"  尺寸: {fig.get_size_inches()[0]:.1f}×{fig.get_size_inches()[1]:.1f} inch")
print(f"  DPI: 300")
print(f"  残留率 = {residual_rate:.1%} (残留{residual:.0f} / 响应幅度{response_amplitude:.0f})")
