# 单文件EDA架构图绘制脚本（优化版：无遮挡+大字号+高清晰）
# 用法: python plot_architecture.py
# 输出: EDATestResult/architecture.png
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import matplotlib.font_manager as fm
import numpy as np
import os

# ===== 字体配置 =====
font_path = 'C:/Windows/Fonts/simhei.ttf'
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# ===== 颜色方案（保留原配色）=====
C_LAYER_BG = {
    'input':    '#E3F2FD',  # 浅蓝
    'parse':    '#E8F5E9',  # 浅绿
    'env':      '#FFF3E0',  # 浅橙
    'core':     '#F3E5F5',  # 浅紫
    'score':    '#FFEBEE',  # 浅红
    'output':   '#E0F2F1',  # 浅青
}
C_LAYER_BORDER = {
    'input':    '#1565C0',
    'parse':    '#2E7D32',
    'env':      '#E65100',
    'core':     '#7B1FA2',
    'score':    '#C62828',
    'output':   '#00695C',
}
C_SUB_BOX = '#FFFFFF'
C_ARROW = '#37474F'
C_TITLE_TEXT = '#FFFFFF'

# ===== 辅助函数（优化字号与行高）=====
def draw_layer_box(ax, x, y, w, h, color_bg, color_border, title, title_color='white'):
    """绘制大层框"""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15", linewidth=2.5,
                         facecolor=color_bg, edgecolor=color_border, zorder=1)
    ax.add_patch(box)
    # 标题标签
    title_box = FancyBboxPatch((x + 0.15, y + h - 0.85), w - 0.3, 0.75,
                               boxstyle="round,pad=0.08", linewidth=0,
                               facecolor=color_border, edgecolor='none', zorder=2)
    ax.add_patch(title_box)
    ax.text(x + w/2, y + h - 0.48, title, ha='center', va='center',
            fontsize=13, fontweight='bold', color=title_color, zorder=3)

def draw_sub_box(ax, x, y, w, h, text, fontsize=9.5, color='white', edgecolor='#546E7A',
                 bold=False, text_color='#263238'):
    """绘制子功能框（优化行高与字号）"""
    weight = 'bold' if bold else 'normal'
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.1", linewidth=1.2,
                         facecolor=color, edgecolor=edgecolor, zorder=4)
    ax.add_patch(box)
    # 支持多行文本，优化行高
    lines = text.split('\n')
    line_h = min(h / (len(lines) + 0.8), fontsize * 0.05)
    total_h = line_h * len(lines)
    start_y = y + h / 2 + total_h / 2 - line_h / 2
    for i, line in enumerate(lines):
        ax.text(x + w/2, start_y - i * line_h, line, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=text_color, zorder=5)

def draw_arrow_down(ax, x, y_top, y_bottom, color='#546E7A'):
    """绘制层间主箭头"""
    ax.annotate('', xy=(x, y_bottom), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                               connectionstyle='arc3,rad=0'), zorder=2)

def draw_sub_arrow(ax, x, y_top, y_bottom, color='#78909C'):
    """绘制层内细箭头（精简使用）"""
    ax.annotate('', xy=(x, y_bottom + 0.05), xytext=(x, y_top - 0.05),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                               connectionstyle='arc3,rad=0'), zorder=3)

# ===== 主图（扩容画布+重新排布）=====
fig, ax = plt.subplots(1, 1, figsize=(36, 50))
ax.set_xlim(0, 36)
ax.set_ylim(0, 50)
ax.axis('off')
ax.set_aspect('equal')

# 总标题（放大字号）
ax.text(18, 49.2, '单文件 EDA（single.py）完整架构', ha='center', va='center',
        fontsize=22, fontweight='bold', color='#1A237E')
ax.text(18, 48.3, '按代码执行顺序分层', ha='center', va='center',
        fontsize=13, color='#546E7A')

# ============================================================
# 第一层：输入层
# ============================================================
LY1_TOP = 47.5
LY1_H = 1.0
draw_layer_box(ax, 1, LY1_TOP - LY1_H, 34, LY1_H,
               C_LAYER_BG['input'], C_LAYER_BORDER['input'], '一、输入层 (Input)')
draw_sub_box(ax, 3, LY1_TOP - 0.82, 30, 0.6,
             '单个电子鼻实验原始 txt 数据文件  → 传入 SingleFileEDA(file_path, skip_first_row=True)',
             fontsize=10, edgecolor=C_LAYER_BORDER['input'])

# ============================================================
# 第二层：初始化与解析预处理层
# ============================================================
LY2_TOP = 46.0
LY2_H = 5.0
draw_layer_box(ax, 1, LY2_TOP - LY2_H, 34, LY2_H,
               C_LAYER_BG['parse'], C_LAYER_BORDER['parse'], '二、初始化与解析预处理层 (parse_file)')

# 子模块 - 第1行
draw_sub_box(ax, 2, LY2_TOP - 1.0, 15, 0.8,
             '版本自动识别：检测数据列数 → 匹配 37/49 传感器硬件版本',
             fontsize=9, edgecolor=C_LAYER_BORDER['parse'])
draw_sub_box(ax, 19, LY2_TOP - 1.0, 15, 0.8,
             '元数据提取：正则匹配日期 + 解析气体类型/浓度/平行样编号',
             fontsize=9, edgecolor=C_LAYER_BORDER['parse'])

# 子模块 - 第2行
draw_sub_box(ax, 2, LY2_TOP - 2.1, 15, 0.8,
             '数据通道拆分：传感器数据 | 环境监控数据 | 时间戳 | 阶段标签列',
             fontsize=9, edgecolor=C_LAYER_BORDER['parse'])
draw_sub_box(ax, 19, LY2_TOP - 2.1, 15, 0.8,
             '阶段自动识别：按标签拆分 P13 / P14 / P15 / P16 / P17 / P18',
             fontsize=9, edgecolor=C_LAYER_BORDER['parse'])

# 子模块 - 第3行 (居中)
draw_sub_box(ax, 7.5, LY2_TOP - 3.2, 21, 0.7,
             '首行零值同步裁切：每个阶段前 N 行核心传感器全零行剔除 → 时间戳与索引同步裁切',
             fontsize=9, edgecolor=C_LAYER_BORDER['parse'])

# 层间箭头
draw_arrow_down(ax, 18, LY1_TOP - LY1_H, LY2_TOP)

# ============================================================
# 第三层：硬件环境体检层
# ============================================================
LY3_TOP = 40.5
LY3_H = 2.7
draw_layer_box(ax, 1, LY3_TOP - LY3_H, 34, LY3_H,
               C_LAYER_BG['env'], C_LAYER_BORDER['env'], '三、全局硬件环境体检层 (analyze_environment)')

draw_sub_box(ax, 2, LY3_TOP - 1.0, 7.5, 0.75,
             '供电电压校验\n均值范围 + 波动幅度', fontsize=8.5, edgecolor=C_LAYER_BORDER['env'])
draw_sub_box(ax, 10.5, LY3_TOP - 1.0, 7.5, 0.75,
             '温度稳定性校验\n均值范围 + 漂移幅度', fontsize=8.5, edgecolor=C_LAYER_BORDER['env'])
draw_sub_box(ax, 19, LY3_TOP - 1.0, 7.5, 0.75,
             '湿度稳定性校验\n均值范围 + 漂移幅度', fontsize=8.5, edgecolor=C_LAYER_BORDER['env'])
draw_sub_box(ax, 27.5, LY3_TOP - 1.0, 7.5, 0.75,
             '气压稳定性校验\n均值范围 + 波动幅度', fontsize=8.5, edgecolor=C_LAYER_BORDER['env'])

draw_sub_box(ax, 9, LY3_TOP - 2.1, 20, 0.6,
             '异常标记 → 不达标项写入警告列表，报告置顶显示 [优先检查]',
             fontsize=9.5, edgecolor=C_LAYER_BORDER['env'], bold=True)

draw_arrow_down(ax, 18, LY2_TOP - LY2_H, LY3_TOP)

# ============================================================
# 第四层：核心分析层（扩容，给足高度）
# ============================================================
LY4_TOP = 37.3
LY4_H = 24.0
draw_layer_box(ax, 1, LY4_TOP - LY4_H, 34, LY4_H,
               C_LAYER_BG['core'], C_LAYER_BORDER['core'], '四、核心分析层 (按执行顺序)')

# ---- (一) 基线稳定性分析 ----
SEC1_TOP = LY4_TOP - 0.6
SEC1_H = 5.3
sec1_color = '#E1BEE7'
draw_sub_box(ax, 2, SEC1_TOP - SEC1_H, 32, SEC1_H,
             '', fontsize=8, edgecolor='#7B1FA2', color='#F5F0FF')
ax.text(2.5, SEC1_TOP - 0.55, '(一) 基线稳定性分析 (analyze_baseline)',
        fontsize=11, fontweight='bold', color='#6A1B9A', zorder=5)

bx = 2.5
draw_sub_box(ax, bx, SEC1_TOP - 1.3, 10, 0.6,
             '基线择优: P13四维初判 → 超标则降级P14', fontsize=8.5, edgecolor='#AB47BC')
draw_sub_box(ax, bx + 10.8, SEC1_TOP - 1.3, 10, 0.6,
             '四维融合判定: A.大均值CV B.小均值STD兜底\nC.全局STD红线 D.极差漂移', fontsize=8, edgecolor='#AB47BC')
draw_sub_box(ax, bx + 21.6, SEC1_TOP - 1.3, 10, 0.6,
             '离群点毛刺检测: 3×IQR边界 → 异常率>5%标记', fontsize=8.5, edgecolor='#AB47BC')

draw_sub_box(ax, bx, SEC1_TOP - 2.2, 10, 0.6,
             '评分: 稳定达标 +20', fontsize=8.5, edgecolor='#7B1FA2', bold=True,
             color='#E8F5E9')
draw_sub_box(ax, bx + 10.8, SEC1_TOP - 2.2, 10, 0.6,
             '评分: 不稳定超标 -15', fontsize=8.5, edgecolor='#7B1FA2', bold=True,
             color='#FFEBEE')
draw_sub_box(ax, bx + 21.6, SEC1_TOP - 2.2, 10, 0.6,
             '评分: 毛刺超标 -10 (全局仅扣一次)', fontsize=8.5, edgecolor='#7B1FA2', bold=True,
             color='#FFF8E1')

draw_sub_box(ax, 7.5, SEC1_TOP - 3.2, 21, 0.5,
             '输出统计量: 均值 | 标准差 | 方差 | 变异系数CV | 极差PTP | IQR边界',
             fontsize=8.5, edgecolor='#CE93D8', color='#FAFAFA')

# ---- (二) 进样阶段分析 ----
SEC2_TOP = SEC1_TOP - SEC1_H - 0.7
SEC2_H = 5.6
draw_sub_box(ax, 2, SEC2_TOP - SEC2_H, 32, SEC2_H,
             '', fontsize=8, edgecolor='#7B1FA2', color='#F5F0FF')
ax.text(2.5, SEC2_TOP - 0.55, '(二) 进样阶段分析 (analyze_sampling)',
        fontsize=11, fontweight='bold', color='#6A1B9A', zorder=5)

draw_sub_box(ax, 2.5, SEC2_TOP - 1.3, 10, 0.6,
             '阶段自动匹配: P15优先 → P17回退', fontsize=8.5, edgecolor='#AB47BC')
draw_sub_box(ax, 13.3, SEC2_TOP - 1.3, 10, 0.6,
             '稳态值提取: 滑动窗口波动校验\n+ 后续差分校验，防伪稳态', fontsize=8, edgecolor='#AB47BC')
draw_sub_box(ax, 24.1, SEC2_TOP - 1.3, 10, 0.6,
             '响应量计算: 绝对响应增量\n+ 归一化响应比率', fontsize=8, edgecolor='#AB47BC')

draw_sub_box(ax, 2.5, SEC2_TOP - 2.2, 15, 0.6,
             '有效响应判定: 响应比率 > 0.1 → 有效响应传感器', fontsize=8.5, edgecolor='#AB47BC')
draw_sub_box(ax, 18.5, SEC2_TOP - 2.2, 15.5, 0.6,
             '评分: 有效响应≥10 +25 / <10 -15', fontsize=8.5, edgecolor='#7B1FA2', bold=True,
             color='#E8F5E9')

draw_sub_box(ax, 5, SEC2_TOP - 3.1, 25, 0.6,
             'P15+P17 同时存在 → 双次进样一致性校验: 皮尔逊 r>0.95 +10 / r<0.7 -10',
             fontsize=8.5, edgecolor='#7B1FA2', bold=True, color='#E3F2FD')

# ---- (三) 时间序列相关性分析 ----
SEC3_TOP = SEC2_TOP - SEC2_H - 0.7
SEC3_H = 3.4
draw_sub_box(ax, 2, SEC3_TOP - SEC3_H, 32, SEC3_H,
             '', fontsize=8, edgecolor='#7B1FA2', color='#F5F0FF')
ax.text(2.5, SEC3_TOP - 0.55, '(三) 时间序列相关性分析 (calculate_temporal_correlation)',
        fontsize=11, fontweight='bold', color='#6A1B9A', zorder=5)

draw_sub_box(ax, 2.5, SEC3_TOP - 1.4, 10, 0.6,
             '数据截取: 进样阶段前50%动态爬坡数据', fontsize=8.5, edgecolor='#AB47BC')
draw_sub_box(ax, 13.3, SEC3_TOP - 1.4, 10, 0.6,
             '矩阵计算: 全传感器两两皮尔逊相关系数', fontsize=8.5, edgecolor='#AB47BC')
draw_sub_box(ax, 24.1, SEC3_TOP - 1.4, 10, 0.6,
             '筛选输出: |r|>0.8 高相关对 + 前15排序', fontsize=8.5, edgecolor='#AB47BC')

# ---- (四) 清洗阶段分析 ----
SEC4_TOP = SEC3_TOP - SEC3_H - 0.7
SEC4_H = 6.3
draw_sub_box(ax, 2, SEC4_TOP - SEC4_H, 32, SEC4_H,
             '', fontsize=8, edgecolor='#7B1FA2', color='#F5F0FF')
ax.text(2.5, SEC4_TOP - 0.55, '(四) 清洗阶段分析 (analyze_cleaning) [重构模块]',
        fontsize=11, fontweight='bold', color='#6A1B9A', zorder=5)

draw_sub_box(ax, 2.5, SEC4_TOP - 1.3, 10, 0.6,
             '阶段匹配: P15→P16(无则P18)\nP17→P18', fontsize=8, edgecolor='#AB47BC')
draw_sub_box(ax, 13.3, SEC4_TOP - 1.3, 10, 0.6,
             '清洗稳态: 末端向前反向搜索\n窗口波动 < 基线5%', fontsize=8, edgecolor='#AB47BC')
draw_sub_box(ax, 24.1, SEC4_TOP - 1.3, 10, 0.6,
             '残留率: 统一取绝对值，恒≥0\n数学定义自洽', fontsize=8, edgecolor='#AB47BC')

# 反向越界子模块
draw_sub_box(ax, 2.5, SEC4_TOP - 2.2, 15, 0.6,
             '反向越界检测: resp_amp × resid < 0 → 越过基线', fontsize=8.5,
             edgecolor='#E53935', color='#FFEBEE', bold=True)
draw_sub_box(ax, 18.5, SEC4_TOP - 2.2, 15.5, 0.6,
             '二级分类: 残留率≤50%→过度清洗 | >50%→基线漂移警告', fontsize=8.5,
             edgecolor='#E53935', color='#FFEBEE', bold=True)

draw_sub_box(ax, 2.5, SEC4_TOP - 3.1, 10, 0.6,
             '优良率统计(方案A): 排除越界传感器\n优秀<5% | 良好5-15% | 较差≥25%', fontsize=8,
             edgecolor='#AB47BC')
draw_sub_box(ax, 13.3, SEC4_TOP - 3.1, 10, 0.6,
             '评分: 优良率≥80% +10\n平均残留≤15% +5\n否则 -10', fontsize=8,
             edgecolor='#7B1FA2', bold=True, color='#E8F5E9')
draw_sub_box(ax, 24.1, SEC4_TOP - 3.1, 10, 0.6,
             '越界占比>15% → 硬件状态警告\n(暂不扣分)', fontsize=8,
             edgecolor='#E53935', color='#FFF8E1', bold=True)

# 层间主箭头
draw_arrow_down(ax, 18, LY3_TOP - LY3_H, LY4_TOP)

# ============================================================
# 第五层：综合质量评分层
# ============================================================
LY5_TOP = 12.8
LY5_H = 2.6
draw_layer_box(ax, 1, LY5_TOP - LY5_H, 34, LY5_H,
               C_LAYER_BG['score'], C_LAYER_BORDER['score'], '五、综合质量评分层 (calculate_quality_score)')

draw_sub_box(ax, 2, LY5_TOP - 1.0, 10, 0.75,
             '基础分: 50', fontsize=10, edgecolor=C_LAYER_BORDER['score'], bold=True)
draw_sub_box(ax, 13, LY5_TOP - 1.0, 10, 0.75,
             '分项累加: 基线 + 进样 + 清洗\n+ 一致性 + 毛刺 等', fontsize=9, edgecolor=C_LAYER_BORDER['score'])
draw_sub_box(ax, 24, LY5_TOP - 1.0, 10, 0.75,
             '范围截断: clamp(0, 100)', fontsize=10, edgecolor=C_LAYER_BORDER['score'], bold=True)

draw_sub_box(ax, 7, LY5_TOP - 2.0, 20, 0.65,
             '评价分级: ≥80 优秀 (建议保留)  |  60-79 良好 (注意警告)  |  <60 较差 (人工复查)',
             fontsize=10, edgecolor=C_LAYER_BORDER['score'], bold=True,
             color='#FFF8E1')

draw_arrow_down(ax, 18, LY4_TOP - LY4_H, LY5_TOP)

# ============================================================
# 第六层：结果输出层
# ============================================================
LY6_TOP = 9.7
LY6_H = 5.2
draw_layer_box(ax, 1, LY6_TOP - LY6_H, 34, LY6_H,
               C_LAYER_BG['output'], C_LAYER_BORDER['output'], '六、结果输出层 (Output)')

# 文本报告
draw_sub_box(ax, 2, LY6_TOP - 1.2, 15, 1.2,
             '文本报告 (save_text_report)\n质量评估报告.txt\n含全模块明细、异常警告、评分与建议',
             fontsize=9.5, edgecolor=C_LAYER_BORDER['output'], bold=True)

# 可视化图表
draw_sub_box(ax, 19, LY6_TOP - 1.2, 15, 3.6,
             '', fontsize=8, edgecolor=C_LAYER_BORDER['output'], color='#FAFAFA')
ax.text(26.5, LY6_TOP - 0.7, '可视化图表 (visualize_all)', fontsize=10.5, fontweight='bold',
        color='#00695C', ha='center', zorder=5)

charts = [
    '① 完整时间序列图: 全传感器 + 分阶段着色标注',
    '② 基线稳定性箱线图: 3×IQR离群点红色实心圆标记',
    '③ 响应增量柱状图: 正响应绿色 / 负响应红色',
    '④ 时间序列相关系数热力图',
    '⑤ 雷达对比图: 基线→进样→清洗 归一化对比',
]
for i, ch in enumerate(charts):
    ypos = LY6_TOP - 1.4 - i * 0.62
    draw_sub_box(ax, 19.3, ypos - 0.42, 14.4, 0.52, ch,
                 fontsize=8.5, edgecolor='#80CBC4', color='white')

draw_arrow_down(ax, 18, LY5_TOP - LY5_H, LY6_TOP)

# ============================================================
# 执行流程总览 + 图例（底部下移，避免拥挤）
# ============================================================
BOTTOM_Y = 2.8
ax.text(18, BOTTOM_Y, '执行流程 (run_full_analysis)',
        fontsize=14, fontweight='bold', color='#1A237E', ha='center')
flow_text = ('parse_file  →  analyze_environment  →  analyze_baseline  →  analyze_sampling  →  '
             'calculate_temporal_correlation  →  analyze_cleaning  →  calculate_quality_score  →  '
             'generate_report  →  visualize_all')
ax.text(18, BOTTOM_Y - 0.7, flow_text,
        fontsize=9.5, color='#37474F', ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ECEFF1', edgecolor='#90A4AE', alpha=0.9))

# 图例
legend_y = BOTTOM_Y - 1.7
legend_items = [
    ('■ 主处理层', C_LAYER_BG['parse'], C_LAYER_BORDER['parse']),
    ('■ 独立功能模块', C_LAYER_BG['env'], C_LAYER_BORDER['env']),
    ('■ 核心分析组', C_LAYER_BG['core'], C_LAYER_BORDER['core']),
    ('■ 评分/输出层', C_LAYER_BG['score'], C_LAYER_BORDER['score']),
    ('□ 子功能块', '#FFFFFF', '#546E7A'),
    ('□ 关键/重构模块', '#FFEBEE', '#E53935'),
]
for i, (label, bg, border) in enumerate(legend_items):
    lx = 4 + i * 5
    box = FancyBboxPatch((lx, legend_y - 0.18), 0.45, 0.35,
                         boxstyle="round,pad=0.03", linewidth=1.2,
                         facecolor=bg, edgecolor=border, zorder=5)
    ax.add_patch(box)
    ax.text(lx + 0.65, legend_y, label, fontsize=9, color='#37474F', va='center')

# 保存
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'EDATestResult')
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'architecture.png')
plt.tight_layout(pad=0.5)
plt.savefig(save_path, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'[OK] 优化版架构图已保存至: {save_path}')
print(f'      尺寸: 36×50 英寸 @ 200 DPI')
print(f'      优化: 无模块遮挡 + 大字号 + 高清晰度')