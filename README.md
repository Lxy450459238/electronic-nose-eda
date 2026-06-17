# 电子鼻 EDA 分析系统

电子鼻传感器数据的探索性数据分析（EDA）工具。支持旧版（37传感器）和新版（49传感器）硬件自动识别。

> GitHub: [https://github.com/Lxy450459238/electronic-nose-eda](https://github.com/Lxy450459238/electronic-nose-eda)

---

## 一键复现指南

### 第 1 步：获取代码

```bash
git clone https://github.com/Lxy450459238/electronic-nose-eda.git
cd electronic-nose-eda
```

如果网络不通，也可以直接下载 ZIP 压缩包解压。

### 第 2 步：一键安装环境

确保电脑已安装 Python 3.8 及以上版本，然后在项目目录下执行：

```bash
pip install -r requirements.txt
```

这会自动安装 `numpy`、`pandas`、`matplotlib`、`seaborn`、`scipy` 五个依赖包，无需逐个手动安装。

### 第 3 步：准备数据

将待分析的 `.txt` 数据文件放到任意目录下。数据需满足：

- 空格分隔的文本文件
- 第 5 列为阶段标签（P13 / P14 / P15 / P16 / P17 / P18）
- 旧版电子鼻（37 传感器）：至少 51 列
- 新版电子鼻（49 传感器）：至少 83 列
- 文件名建议遵循 `日期-编号-气体-浓度-重复次数.txt` 格式

### 第 4 步：运行分析

**分析单个文件：**

```bash
python single.py 数据文件路径
```

示例：

```bash
python single.py D:\data\20230320-302-bingtong-30ppm-1.txt
```

输出 5 张图 + 1 份文本报告，保存在 `EDATestResult/` 目录下。

**批量分析一个文件夹中的所有文件：**

```bash
python multi.py 数据文件夹路径
```

示例：

```bash
python multi.py D:\data\多样性样本2
```

输出 6 张图 + 1 份文本报告，保存在 `EDATestResult/` 目录下。

### 第 5 步：查看结果

打开项目目录下的 `EDATestResult/` 文件夹，包含：

| 单文件输出 | 多文件输出 |
|-----------|-----------|
| `*_timeseries.png` — 全周期时间序列 | `01_boxplot.png` — 各传感器响应分布 |
| `*_baseline_boxplot.png` — 基线箱线图 | `02_timeseries.png` — 时间序列采样 |
| `*_response_delta.png` — 响应增量图 | `03_histogram.png` — 响应直方图 |
| `*_temporal_correlation.png` — 相关热力图 | `04_correlation_heatmap.png` — 相关系数矩阵 |
| `*_radar_comparison.png` — 雷达对比图 | `05_*_radar.png` — 雷达图 |
| `*_report.txt` — 文本评估报告 | `06_*.png` — 稳定性表/区分度排名 |
| | `multi_file_report.txt` — 文本评估报告 |

### 常见问题

| 问题 | 解决方法 |
|------|---------|
| 报错 `找不到文件` | 用引号包裹含括号/空格的路径，或使用绝对路径 |
| 报错 `文件夹中没有找到txt文件` | 确保文件夹路径正确，且内部有 `.txt` 文件 |
| 报错 `数据文件列数不足` | 检查数据格式：旧版需 ≥51 列，新版需 ≥83 列 |
| 中文图表乱码 | 确保 `C:\Windows\Fonts\simhei.ttf`（黑体）存在 |

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `config.py` | 通用配置，包含传感器映射、分析参数、可视化参数等 |
| `single.py` | 单文件分析：对一个样本进行完整的质量评估 |
| `multi.py` | 多文件分析：批量对比多个样本的稳定性/区分度 |
| `算法说明书.docx` | 算法原理详细文档 |
| `修改细节.md` | 整理过程中的所有修改记录 |

## 数据格式要求

## 输出目录

所有结果统一输出到脚本所在目录下的 `EDATestResult/` 文件夹。
每次运行前自动清空旧结果，避免数据干扰。

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| EDA(6.8) | 2026.06.08 | 最终整理版：统一命名、清理冗余代码、v6修复并入(single+multi同步) |

## 算法参考

详见 `算法说明书.docx`。
