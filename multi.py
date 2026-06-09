# 电子鼻多文件EDA分析 - 自动识别新旧版本
# 修改时间: 2026.06.08
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import iqr, f_oneway
from itertools import combinations
import os
import sys
import shutil
import glob
import warnings
import re

from config import UniversalConfig as Config

warnings.filterwarnings('ignore')

# ===== 字体配置（解决中文显示问题）=====
font_path = 'C:/Windows/Fonts/simhei.ttf'
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False

print(f"使用字体: {prop.get_name()}")


class MultiFileEDA:
    """多文件探索性数据分析"""

    def __init__(self, file_paths):
        """
        初始化

        参数:
            file_paths: list, txt文件路径列表
        """
        self.file_paths = file_paths
        self.n_files = len(file_paths)
        self.P15_matrix = None
        self.P17_matrix = None
        self.file_metadata = []
        self.sample_diversity = None
        self.results = {}
        self.outlier_samples = []
        self.env_warnings = []           # 环境通道异常汇总

        self.version_config = None
        self.num_sensors = None

        print(f"{'=' * 60}")
        print(f"多文件EDA分析")
        print(f"文件数量: {self.n_files}")
        print(f"{'=' * 60}")

    def load_and_extract_data(self):
        """加载所有文件并提取稳态数据"""
        print("\n正在加载文件并提取稳态数据...")

        P15_list = []
        P17_list = []

        for i, file_path in enumerate(self.file_paths):
            try:
                filename = os.path.basename(file_path)

                # 解析文件名元数据
                metadata = self._parse_filename(filename)
                metadata['file_path'] = file_path
                metadata['file_index'] = i
                self.file_metadata.append(metadata)

                # 读取数据
                data = pd.read_csv(file_path, sep=' ', header=None,
                                   encoding='utf-8', on_bad_lines='skip')

                # 第一个文件检测版本
                if i == 0:
                    data_columns = data.shape[1]
                    self.version_config = Config.detect_version(data_columns)
                    self.num_sensors = self.version_config['sensor_count']
                    print(f"[OK] 检测到{self.version_config['name']}")
                    print(f"[OK] 传感器数量: {self.num_sensors}个\n")

                stage_column = data.iloc[:, 4]
                sensor_data = Config.extract_sensor_data(data, self.version_config)

                # 首行零值跳过：检查前N行核心传感器是否含0（硬件通信未就绪）
                check_cols = min(Config.DATA_SKIP_CHECK_COLS, self.num_sensors)
                valid_start = 0
                for r in range(min(Config.DATA_SKIP_ROWS_MAX, len(sensor_data))):
                    if np.any(sensor_data[r, :check_cols] == 0):
                        valid_start = r + 1
                    else:
                        break
                if valid_start > 0:
                    sensor_data = sensor_data[valid_start:]
                    stage_column = stage_column[valid_start:]

                # 提取环境监控数据（仅第一个文件输出日志）
                if i == 0:
                    try:
                        raw_env = Config.extract_env_data(data, self.version_config)
                        if raw_env is not None:
                            self.env_data_sample = raw_env.astype(float)
                    except Exception:
                        self.env_data_sample = None

                # 提取P15稳态
                if (stage_column == 'P15').any():
                    P15_data = sensor_data[stage_column == 'P15']
                    P15_steady = self._extract_steady_state(P15_data)
                    P15_list.append(P15_steady)
                else:
                    print(f"  [WARNING] 文件 {filename} 缺少P15数据")
                    P15_list.append(np.full(self.num_sensors, np.nan))

                # 提取P17稳态（如果有）
                if (stage_column == 'P17').any():
                    P17_data = sensor_data[stage_column == 'P17']
                    P17_steady = self._extract_steady_state(P17_data)
                    P17_list.append(P17_steady)
                else:
                    P17_list.append(np.full(self.num_sensors, np.nan))

                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i + 1}/{self.n_files} 个文件")

            except Exception as e:
                print(f"  [ERROR] 文件 {filename} 处理失败: {str(e)}")
                P15_list.append(np.full(self.num_sensors, np.nan))
                P17_list.append(np.full(self.num_sensors, np.nan))

        # 转换为矩阵
        self.P15_matrix = np.array(P15_list)
        self.P17_matrix = np.array(P17_list)

        # 移除全NaN的样本
        valid_mask = ~np.all(np.isnan(self.P15_matrix), axis=1)
        self.P15_matrix = self.P15_matrix[valid_mask]
        self.file_metadata = [m for m, v in zip(self.file_metadata, valid_mask) if v]

        print(f"\n[OK] 成功加载 {len(self.P15_matrix)} 个有效样本")

        return self

    def _parse_filename(self, filename):
        """从文件名提取元数据 - 智能日期提取（正则匹配8位连续数字）"""
        name_no_ext = filename.replace('.txt', '')
        parts = name_no_ext.split('-')

        # 智能提取日期：寻找连续的8位以上数字 (如 20211013 或 20230320)
        date_match = re.search(r'20\d{6}', name_no_ext)
        date_str = date_match.group(0) if date_match else 'unknown'

        gas_type = 'unknown'
        concentration = 'unknown'

        if len(parts) >= 4:
            if parts[0] == date_str:
                gas_type = parts[2] if len(parts) > 2 else 'unknown'
                concentration = parts[3] if len(parts) > 3 else 'unknown'
            else:
                gas_type = parts[2] if len(parts) > 2 else 'unknown'
                concentration = parts[3] if len(parts) > 3 else 'unknown'

        return {
            'filename': filename,
            'date': date_str,
            'id': parts[1] if len(parts) > 1 else 'unknown',
            'gas_type': gas_type,
            'concentration': concentration,
            'replicate': parts[4] if len(parts) > 4 else '1'
        }

    def _extract_steady_state(self, stage_data):
        """提取稳态均值（简化版）"""
        steady_values = []

        for sensor_idx in range(self.num_sensors):
            sensor_series = stage_data[:, sensor_idx]

            if len(sensor_series) < 30:
                steady_values.append(np.mean(sensor_series))
                continue

            # 使用后50%数据作为稳态
            steady_segment = sensor_series[len(sensor_series) // 2:]
            steady_values.append(np.mean(steady_segment))

        return np.array(steady_values)

    def analyze_environment(self):
        """全局环境与系统通道体检（基于第一个样本数据）"""
        print("\n--- 全局硬件健康体检 ---")
        if not hasattr(self, 'env_data_sample') or self.env_data_sample is None:
            print("  [INFO] 未配置环境通道，跳过体检。")
            return self

        env_data = self.env_data_sample
        env_mean = np.nanmean(env_data, axis=0)
        env_std = np.nanstd(env_data, axis=0)
        env_max = np.nanmax(env_data, axis=0)
        env_min = np.nanmin(env_data, axis=0)

        env_name_map = self.version_config.get('env_name_map', {})

        for i in range(env_data.shape[1]):
            name = env_name_map.get(i, f"Env_{i}")
            mean_val = env_mean[i]
            std_val = env_std[i]

            is_abnormal = False
            reason = ""

            # 1. 供电电压校验
            if 'VCC' in name.upper():
                min_v, max_v = Config.ENV_VALID_RANGES['VCC5V']
                if not (min_v <= mean_val <= max_v):
                    is_abnormal = True
                    reason = f"均值越界 ({mean_val:.1f}mV 不在 {min_v}~{max_v} 内)"
                elif std_val > 150.0:
                    is_abnormal = True
                    reason = f"全局波动剧烈 (STD={std_val:.1f}mV)"

            # 2. 温度校验
            elif '温度' in name or 'TEMPERATURE' in name.upper():
                min_v, max_v = Config.ENV_VALID_RANGES['TEMP']
                if not (min_v <= mean_val <= max_v):
                    is_abnormal = True
                    reason = f"均值越界 ({mean_val:.1f}℃ 不在 {min_v}~{max_v} 内)"
                elif std_val > 10.0:
                    is_abnormal = True
                    reason = f"全局温度漂移 (STD={std_val:.2f}℃, 范围:{env_min[i]:.1f}~{env_max[i]:.1f}℃)"

            # 3. 湿度校验
            elif '湿度' in name or 'HUMIDITY' in name.upper():
                min_v, max_v = Config.ENV_VALID_RANGES['HUM']
                if not (min_v <= mean_val <= max_v):
                    is_abnormal = True
                    reason = f"均值越界 ({mean_val:.1f}% 不在 {min_v}~{max_v} 内)"
                elif std_val > 25.0:
                    is_abnormal = True
                    reason = f"全局湿度漂移 (STD={std_val:.2f}%)"

            # 4. 压力校验
            elif '压力' in name or 'PRESS' in name.upper():
                min_v, max_v = Config.ENV_VALID_RANGES['PRESS']
                if not (min_v <= mean_val <= max_v):
                    is_abnormal = True
                    reason = f"均值越界 ({mean_val:.1f}mbar 不在 {min_v}~{max_v} 内)"
                elif std_val > 50.0:
                    is_abnormal = True
                    reason = f"气压波动过大 (STD={std_val:.1f}mbar)"

            if is_abnormal:
                self.env_warnings.append((name, reason))

        self.results['environment'] = self.env_warnings

        if self.env_warnings:
            print(f"  [WARNING] 硬件体检异常: {len(self.env_warnings)} 项环境指标未达标")
            for name, reason in self.env_warnings:
                print(f"    [!] {name}: {reason}")
        else:
            print("  [OK] 所有供电、温度、湿度、气压在整个实验全周期内均保持稳定。")

        return self

    def check_diversity(self):
        """判断样本多样性"""
        print("\n--- 样本多样性判断 ---")

        gas_types = set([m['gas_type'] for m in self.file_metadata])
        concentrations = set([m['concentration'] for m in self.file_metadata])

        print(f"  检测到气体种类: {len(gas_types)} 种 {list(gas_types)}")
        print(f"  检测到浓度水平: {len(concentrations)} 种 {list(concentrations)}")

        if len(gas_types) == 1 and len(concentrations) == 1:
            self.sample_diversity = "低多样性"
            print(f"  判定: 低多样性（同类气体同浓度）")
        else:
            self.sample_diversity = "高多样性"
            print(f"  判定: 高多样性（多类气体或多浓度）")

        return self

    def analyze_statistics(self):
        """均值与方差分析"""
        print("\n--- 均值与方差分析 ---")

        sensor_means = np.mean(self.P15_matrix, axis=0)
        sensor_stds = np.std(self.P15_matrix, axis=0)
        sensor_cvs = sensor_stds / (sensor_means + 1e-6)

        if self.sample_diversity == "低多样性":
            # 低多样性：评估稳定性
            print("  [稳定性评估]")
            unstable_sensors = np.where(sensor_cvs > Config.STABILITY_CV_UNSTABLE)[0]
            stable_sensors = np.where(sensor_cvs < Config.STABILITY_CV_STABLE)[0]

            print(f"    稳定传感器（CV<{Config.STABILITY_CV_STABLE}）: {len(stable_sensors)}个")
            print(f"    不稳定传感器（CV>{Config.STABILITY_CV_UNSTABLE}）: {len(unstable_sensors)}个")
            if len(unstable_sensors) > 0:
                print(f"    不稳定传感器列表: {unstable_sensors.tolist()}")

            self.results['stability'] = {
                'means': sensor_means,
                'stds': sensor_stds,
                'cvs': sensor_cvs,
                'stable_sensors': stable_sensors,
                'unstable_sensors': unstable_sensors
            }

        else:
            # 高多样性：评估区分度
            print("  [区分度评估]")
            discrimination_df = self._calculate_discrimination()

            print(f"\n  区分度排名（前10）:")
            print(discrimination_df.head(10).to_string(index=False))

            self.results['discrimination'] = discrimination_df

        return self

    def _calculate_discrimination(self):
        """计算传感器区分度（高多样性场景）- 自适应归一化方案"""
        from scipy.stats import f_oneway

        labels = np.array([m['gas_type'] + '-' + m['concentration']
                           for m in self.file_metadata])

        results = []
        all_snr_values = []
        all_f_values = []

        # 第一遍：收集所有SNR和F值
        print("\n  [数据收集] 计算所有传感器的SNR和F值...")
        for sensor_idx in range(self.num_sensors):
            sensor_data = self.P15_matrix[:, sensor_idx]

            # 方法1：信噪比（SNR）
            unique_labels = np.unique(labels)
            gas_means = []
            gas_stds = []

            for label in unique_labels:
                mask = labels == label
                gas_means.append(np.mean(sensor_data[mask]))
                gas_stds.append(np.std(sensor_data[mask]))

            max_diff = np.max(gas_means) - np.min(gas_means)
            avg_std = np.mean(gas_stds)
            snr = max_diff / (avg_std + 1e-6)

            # 确保SNR是有效数字
            if np.isnan(snr) or np.isinf(snr):
                snr = 0.0
            all_snr_values.append(snr)

            # 方法2：ANOVA F值
            groups = [sensor_data[labels == label] for label in unique_labels]
            try:
                F_value, p_value = f_oneway(*groups)
                if np.isnan(F_value) or np.isinf(F_value):
                    F_value = 0.0
            except:
                F_value, p_value = 0.0, 1.0
            all_f_values.append(F_value)

        # 自适应归一化：使用95分位数作为归一化因子
        all_snr_values_array = np.array(all_snr_values)
        all_f_values_array = np.array(all_f_values)

        # 过滤掉无效值
        valid_snr = all_snr_values_array[~np.isnan(all_snr_values_array) & ~np.isinf(all_snr_values_array)]
        valid_f = all_f_values_array[~np.isnan(all_f_values_array) & ~np.isinf(all_f_values_array)]

        # 计算95分位数，如果没有有效数据则使用默认值
        if len(valid_snr) > 0:
            snr_factor = max(np.percentile(valid_snr, 95), 1.0)
        else:
            snr_factor = 10.0
            print("  [WARNING] 所有SNR值无效，使用默认归一化因子")

        if len(valid_f) > 0:
            f_factor = max(np.percentile(valid_f, 95), 1.0)
        else:
            f_factor = 50.0
            print("  [WARNING] 所有F值无效，使用默认归一化因子")

        # 打印归一化信息
        print(f"\n  [归一化因子]")
        print(f"    有效数据 - SNR: {len(valid_snr)}/{len(all_snr_values)} 个")
        print(f"    有效数据 - F值: {len(valid_f)}/{len(all_f_values)} 个")
        print(f"    数据范围 - SNR: {np.min(valid_snr):.2f} ~ {np.max(valid_snr):.2f}")
        print(f"    数据范围 - F值: {np.min(valid_f):.2f} ~ {np.max(valid_f):.2f}")
        print(f"    归一化因子 - SNR: {snr_factor:.2f} (95分位数)")
        print(f"    归一化因子 - F值: {f_factor:.2f} (95分位数)")
        print(f"    检测到: {self.version_config['name']}")

        # 第二遍：计算归一化评分
        print(f"\n  [评分计算] 计算传感器区分度评分...")
        for sensor_idx in range(self.num_sensors):
            sensor_data = self.P15_matrix[:, sensor_idx]
            snr = all_snr_values[sensor_idx]

            # 重新计算F值和p值
            unique_labels = np.unique(labels)
            groups = [sensor_data[labels == label] for label in unique_labels]
            try:
                F_value, p_value = f_oneway(*groups)
                if np.isnan(F_value) or np.isinf(F_value):
                    F_value = 0.0
                if np.isnan(p_value) or np.isinf(p_value):
                    p_value = 1.0
            except:
                F_value, p_value = 0.0, 1.0

            # 使用自适应因子归一化
            snr_norm = min(snr / snr_factor, 1.0) if snr_factor > 0 else 0.0
            f_norm = min(F_value / f_factor, 1.0) if f_factor > 0 else 0.0

            # 确保归一化值是有效的
            if np.isnan(snr_norm) or np.isinf(snr_norm):
                snr_norm = 0.0
            if np.isnan(f_norm) or np.isinf(f_norm):
                f_norm = 0.0

            score = Config.DISCRIMINATION_SNR_WEIGHT * snr_norm + Config.DISCRIMINATION_F_WEIGHT * f_norm

            if np.isnan(score) or np.isinf(score):
                score = 0.0

            # 推荐等级判断
            if score > Config.DISCRIMINATION_SCORE_STRONG and p_value < Config.DISCRIMINATION_P_VALUE_STRONG:
                recommendation = "强烈推荐保留"
            elif score > Config.DISCRIMINATION_SCORE_NORMAL and p_value < Config.DISCRIMINATION_P_VALUE_NORMAL:
                recommendation = "推荐保留"
            elif score > Config.DISCRIMINATION_SCORE_ACCEPTABLE:
                recommendation = "可以保留"
            else:
                recommendation = "建议剔除"

            sensor_name = Config.get_sensor_name(sensor_idx, self.version_config)

            results.append({
                'Sensor': sensor_name,
                'SNR': round(snr, 2),
                'F_value': round(F_value, 2),
                'p_value': f'{p_value:.4f}',
                'Score': round(score, 3),
                'Recommendation': recommendation
            })

        df = pd.DataFrame(results)
        df = df.sort_values('Score', ascending=False).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, self.num_sensors + 1))

        # 打印评分统计
        print(f"\n  [评分统计]")
        print(f"    评分范围: {df['Score'].min():.3f} ~ {df['Score'].max():.3f}")
        print(f"    平均评分: {df['Score'].mean():.3f}")
        print(f"    强烈推荐: {len(df[df['Recommendation'] == '强烈推荐保留'])}个")
        print(f"    推荐保留: {len(df[df['Recommendation'] == '推荐保留'])}个")
        print(f"    可以保留: {len(df[df['Recommendation'] == '可以保留'])}个")
        print(f"    建议剔除: {len(df[df['Recommendation'] == '建议剔除'])}个")

        nan_count = df['Score'].isna().sum()
        if nan_count > 0:
            print(f"  [WARNING] 仍有{nan_count}个传感器评分为NaN")

        return df

    def detect_outliers(self):
        """箱线图异常检测"""
        print("\n--- 异常样本检测 ---")

        outlier_counts = np.zeros(len(self.P15_matrix))

        for sensor_idx in range(self.num_sensors):
            sensor_data = self.P15_matrix[:, sensor_idx]

            Q1 = np.percentile(sensor_data, 25)
            Q3 = np.percentile(sensor_data, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - Config.OUTLIER_IQR_MULTIPLIER * IQR
            upper_bound = Q3 + Config.OUTLIER_IQR_MULTIPLIER * IQR

            outliers = (sensor_data < lower_bound) | (sensor_data > upper_bound)
            outlier_counts += outliers

        # 在多个传感器上都异常的样本
        severe_outliers = np.where(outlier_counts > Config.OUTLIER_MIN_ABNORMAL_SENSORS)[0]

        for idx in severe_outliers:
            self.outlier_samples.append({
                'index': idx,
                'filename': self.file_metadata[idx]['filename'],
                'outlier_count': int(outlier_counts[idx]),
                'gas_type': self.file_metadata[idx]['gas_type']
            })

        print(f"  检测到 {len(severe_outliers)} 个异常样本（在>{Config.OUTLIER_MIN_ABNORMAL_SENSORS}个传感器上异常）")
        if len(severe_outliers) > 0:
            print("\n  异常样本列表:")
            for outlier in self.outlier_samples:
                print(f"    {outlier['filename']} (异常传感器数: {outlier['outlier_count']})")

        self.results['outliers'] = self.outlier_samples

        return self

    def calculate_correlation(self):
        """相关系数矩阵分析"""
        print("\n--- 相关系数矩阵分析 ---")

        if self.sample_diversity == "低多样性":
            print("  [时间序列相关性平均]")
            self._calculate_temporal_correlation_average()

        else:
            print("  [稳态相关系数矩阵]")
            corr_matrix = np.corrcoef(self.P15_matrix.T)

            # 找出高度相关的传感器对
            high_corr_pairs = []
            for i in range(self.num_sensors):
                for j in range(i + 1, self.num_sensors):
                    if abs(corr_matrix[i, j]) > Config.CORRELATION_HIGH_THRESHOLD:
                        high_corr_pairs.append((i, j, corr_matrix[i, j]))

            print(f"  发现 {len(high_corr_pairs)} 对高度相关的传感器（|r|>{Config.CORRELATION_HIGH_THRESHOLD}）")
            if len(high_corr_pairs) > 0:
                print("  高相关传感器对（前10个）:")
                for i, j, r in high_corr_pairs[:10]:
                    sensor_i = Config.get_sensor_name(i, self.version_config)
                    sensor_j = Config.get_sensor_name(j, self.version_config)
                    print(f"    {sensor_i} <-> {sensor_j}: r={r:.3f}")

            self.results['steady_correlation'] = {
                'matrix': corr_matrix,
                'high_corr_pairs': high_corr_pairs
            }

        return self

    def _calculate_temporal_correlation_average(self):
        """计算时间序列相关性平均（低多样性场景）"""
        n_samples = min(Config.CORRELATION_SAMPLE_COUNT, len(self.file_paths))
        sample_indices = np.random.choice(len(self.file_paths), n_samples, replace=False)

        corr_matrices = []

        print(f"  正在处理 {n_samples} 个样本...")

        for idx in sample_indices:
            try:
                file_path = self.file_paths[idx]
                data = pd.read_csv(file_path, sep=' ', header=None,
                                   encoding='utf-8', on_bad_lines='skip')

                stage_column = data.iloc[:, 4]
                sensor_data = Config.extract_sensor_data(data, self.version_config)

                if (stage_column == 'P15').any():
                    P15_data = sensor_data[stage_column == 'P15']

                    # 使用前50%数据（动态响应期）
                    dynamic_length = int(len(P15_data) * Config.CORRELATION_DYNAMIC_RATIO)
                    dynamic_data = P15_data[:dynamic_length, :]

                    if dynamic_data.shape[0] > Config.CORRELATION_MIN_POINTS:
                        corr_matrix = np.corrcoef(dynamic_data.T)
                        corr_matrices.append(corr_matrix)

            except Exception as e:
                continue

        if len(corr_matrices) > 0:
            avg_corr_matrix = np.mean(corr_matrices, axis=0)

            high_corr_pairs = []
            for i in range(self.num_sensors):
                for j in range(i + 1, self.num_sensors):
                    if abs(avg_corr_matrix[i, j]) > Config.CORRELATION_HIGH_THRESHOLD:
                        high_corr_pairs.append((i, j, avg_corr_matrix[i, j]))

            print(f"  发现 {len(high_corr_pairs)} 对高度相关的传感器（|r|>{Config.CORRELATION_HIGH_THRESHOLD}）")
            if len(high_corr_pairs) > 0:
                print("  高相关传感器对（前10个）:")
                for i, j, r in high_corr_pairs[:10]:
                    sensor_i = Config.get_sensor_name(i, self.version_config)
                    sensor_j = Config.get_sensor_name(j, self.version_config)
                    print(f"    {sensor_i} <-> {sensor_j}: r={r:.3f}")

            self.results['temporal_correlation'] = {
                'matrix': avg_corr_matrix,
                'high_corr_pairs': high_corr_pairs,
                'n_samples_used': len(corr_matrices)
            }
        else:
            print("  [WARNING] 未能计算时间序列相关性")

    def generate_report(self, save_path):
        """生成文本报告"""
        print(f"\n正在生成文本报告...")

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write("多文件EDA分析报告\n")
            f.write(f"{'=' * 60}\n\n")

            f.write(f"电子鼻版本: {self.version_config['name']}\n")
            f.write(f"传感器数量: {self.num_sensors}个\n")
            f.write(f"样本数量: {len(self.P15_matrix)}\n")
            f.write(f"样本多样性: {self.sample_diversity}\n\n")

            # 气体类型统计
            gas_types = {}
            for m in self.file_metadata:
                key = f"{m['gas_type']}-{m['concentration']}"
                gas_types[key] = gas_types.get(key, 0) + 1

            f.write("样本分布:\n")
            for gas, count in gas_types.items():
                f.write(f"  {gas}: {count}个样本\n")
            f.write("\n")

            # 硬件体检结果
            if 'environment' in self.results:
                f.write("--- 硬件体检 (环境与系统通道) ---\n\n")
                abnormal_env = self.results['environment']
                if abnormal_env:
                    for name, reason in abnormal_env:
                        f.write(f"  [异常] {name}: {reason}\n")
                else:
                    f.write(f"  [OK] 实验全周期内供电、温度、湿度及气压均保持稳定。\n")
                f.write("\n")

            # 传感器分析
            if self.sample_diversity == "低多样性":
                f.write("--- 传感器稳定性分析 ---\n\n")
                stability = self.results['stability']
                f.write(f"稳定传感器（CV<{Config.STABILITY_CV_STABLE}）: {len(stability['stable_sensors'])}个\n")
                f.write(f"不稳定传感器（CV>{Config.STABILITY_CV_UNSTABLE}）: {len(stability['unstable_sensors'])}个\n\n")

                if len(stability['unstable_sensors']) > 0:
                    f.write("不稳定传感器列表:\n")
                    for idx in stability['unstable_sensors']:
                        cv = stability['cvs'][idx]
                        sensor_name = Config.get_sensor_name(idx, self.version_config)
                        f.write(f"  {sensor_name}: CV={cv:.3f}\n")
                    f.write("\n")

            else:
                f.write("--- 传感器区分度分析 ---\n\n")
                disc_df = self.results['discrimination']
                f.write(f"区分度排名（前{Config.REPORT_TOP_SENSORS}）:\n")
                f.write(disc_df.head(Config.REPORT_TOP_SENSORS).to_string(index=False))
                f.write("\n\n")

                recommended = disc_df[disc_df['Recommendation'].str.contains('推荐保留')]
                f.write(f"推荐保留传感器: {len(recommended)}个\n")
                f.write(f"建议剔除传感器: {self.num_sensors - len(recommended)}个\n\n")

            # 异常样本
            f.write("--- 异常样本检测 ---\n\n")
            f.write(f"检测到异常样本: {len(self.outlier_samples)}个\n\n")

            if len(self.outlier_samples) > 0:
                f.write("异常样本列表:\n")
                for outlier in self.outlier_samples:
                    f.write(f"  {outlier['filename']}\n")
                    f.write(f"    异常传感器数: {outlier['outlier_count']}\n")
                    f.write(f"    气体类型: {outlier['gas_type']}\n")
                f.write("\n")

            # 相关性分析
            f.write("--- 传感器相关性分析 ---\n\n")

            if 'temporal_correlation' in self.results:
                corr_info = self.results['temporal_correlation']
                f.write(f"时间序列相关性分析（基于{corr_info['n_samples_used']}个样本）\n")
                f.write(f"高度相关传感器对: {len(corr_info['high_corr_pairs'])}对\n\n")

            elif 'steady_correlation' in self.results:
                corr_info = self.results['steady_correlation']
                f.write(f"稳态相关系数分析\n")
                f.write(f"高度相关传感器对（可能冗余）: {len(corr_info['high_corr_pairs'])}对\n\n")

                if len(corr_info['high_corr_pairs']) > 0:
                    f.write("高相关传感器对列表:\n")
                    for i, j, r in corr_info['high_corr_pairs'][:Config.REPORT_TOP_PAIRS]:
                        sensor_i = Config.get_sensor_name(i, self.version_config)
                        sensor_j = Config.get_sensor_name(j, self.version_config)
                        f.write(f"  {sensor_i} <-> {sensor_j}: r={r:.3f}\n")

        print(f"[OK] 报告已保存至: {save_path}")

    def visualize_all(self, save_dir):
        """生成所有可视化图表"""
        print("\n--- 生成可视化图表 ---")

        # 1. 箱线图
        self.plot_boxplot(os.path.join(save_dir, '01_boxplot.png'))

        # 2. 时间序列图
        if len(self.file_paths) <= 20:
            self.plot_timeseries_sample(os.path.join(save_dir, '02_timeseries.png'))

        # 3. 直方图
        self.plot_histogram(os.path.join(save_dir, '03_histogram.png'))

        # 4. 相关系数热力图
        self.plot_correlation_heatmap(os.path.join(save_dir, '04_correlation_heatmap.png'))

        # 5. 根据样本多样性选择雷达图
        if self.sample_diversity == "低多样性":
            self.plot_standard_radar(os.path.join(save_dir, '05_standard_radar.png'))
            self.plot_stability_table(os.path.join(save_dir, '06_stability_table.png'))
        else:
            self.plot_multi_gas_radar(os.path.join(save_dir, '05_multi_gas_radar.png'))
            self.plot_discrimination_ranking(os.path.join(save_dir, '06_discrimination_ranking.png'))

        print("\n[OK] 所有图表已生成完毕！")

    def plot_boxplot(self, save_path):
        """绘制箱线图"""
        print("\n正在生成箱线图...")

        fig, ax = plt.subplots(figsize=Config.VIS_FIGSIZE_BOXPLOT)

        sensor_labels = [Config.get_sensor_name(i, self.version_config) for i in range(self.num_sensors)]

        bp = ax.boxplot([self.P15_matrix[:, i] for i in range(self.num_sensors)],
                        labels=sensor_labels,
                        patch_artist=True,
                        showfliers=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(Config.VIS_ALPHA_BOX)

        ax.set_xlabel('传感器', fontsize=12)
        ax.set_ylabel('响应值', fontsize=12)
        ax.set_title(f'传感器响应分布箱线图 (N={len(self.P15_matrix)})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=Config.VIS_ALPHA_GRID, axis='y')
        plt.xticks(rotation=90, fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=Config.VIS_DPI, bbox_inches='tight')
        plt.close()

        print(f"箱线图已保存至: {save_path}")

    def plot_timeseries_sample(self, save_path):
        """绘制时间序列采样图"""
        print("\n正在生成时间序列采样图...")

        n_samples = min(Config.CORRELATION_SAMPLE_COUNT, len(self.file_paths))
        sample_indices = np.random.choice(len(self.file_paths), n_samples, replace=False)

        fig, axes = plt.subplots(n_samples, 1, figsize=Config.VIS_FIGSIZE_TIMESERIES, sharex=True)
        if n_samples == 1:
            axes = [axes]

        for idx, file_idx in enumerate(sample_indices):
            try:
                file_path = self.file_paths[file_idx]
                data = pd.read_csv(file_path, sep=' ', header=None,
                                   encoding='utf-8', on_bad_lines='skip')

                stage_column = data.iloc[:, 4]
                sensor_data = Config.extract_sensor_data(data, self.version_config)

                if (stage_column == 'P15').any():
                    P15_data = sensor_data[stage_column == 'P15']

                    axes[idx].plot(P15_data, alpha=Config.VIS_ALPHA_LINE)
                    axes[idx].set_title(f'样本 {idx + 1}: {os.path.basename(file_path)[:30]}...',
                                        fontsize=10)
                    axes[idx].set_ylabel('响应值', fontsize=9)
                    axes[idx].grid(True, alpha=Config.VIS_ALPHA_GRID)

            except Exception as e:
                continue

        axes[-1].set_xlabel('时间点', fontsize=10)

        plt.suptitle('时间序列采样图（P15阶段）', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=Config.VIS_DPI, bbox_inches='tight')
        plt.close()

        print(f"时间序列图已保存至: {save_path}")

    def plot_histogram(self, save_path):
        """绘制传感器响应直方图"""
        print("\n正在生成响应直方图...")

        n_cols = 5
        n_rows = int(np.ceil(self.num_sensors / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=Config.VIS_FIGSIZE_HISTOGRAM_MULTI)

        for i in range(self.num_sensors):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            sensor_name = Config.get_sensor_name(i, self.version_config)
            ax.hist(self.P15_matrix[:, i], bins=Config.VIS_HIST_BINS,
                    color='steelblue', alpha=Config.VIS_ALPHA_BOX, edgecolor='black')
            ax.set_title(sensor_name, fontsize=9)
            ax.set_xlabel('响应值', fontsize=8)
            ax.set_ylabel('频数', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=Config.VIS_ALPHA_GRID)

        # 隐藏多余的子图
        for i in range(self.num_sensors, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.axis('off')

        plt.suptitle('传感器响应分布直方图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=Config.VIS_DPI, bbox_inches='tight')
        plt.close()

        print(f"直方图已保存至: {save_path}")

    def plot_correlation_heatmap(self, save_path):
        """绘制相关系数热力图"""
        print("\n正在生成相关系数热力图...")

        # 获取相关系数矩阵
        if 'temporal_correlation' in self.results:
            corr_matrix = self.results['temporal_correlation']['matrix']
            title_suffix = '(时间序列平均)'
        elif 'steady_correlation' in self.results:
            corr_matrix = self.results['steady_correlation']['matrix']
            title_suffix = '(稳态数据)'
        else:
            corr_matrix = np.corrcoef(self.P15_matrix.T)
            title_suffix = '(稳态数据)'

        fig, ax = plt.subplots(figsize=Config.VIS_FIGSIZE_CORRELATION)

        sensor_labels = [Config.get_sensor_name(i, self.version_config) for i in range(self.num_sensors)]

        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(range(self.num_sensors))
        ax.set_yticks(range(self.num_sensors))
        ax.set_xticklabels(sensor_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(sensor_labels, fontsize=7)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('相关系数', fontsize=10)

        ax.set_title(f'传感器相关系数热力图 {title_suffix}',
                     fontsize=14, fontweight='bold', pad=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=Config.VIS_DPI_HIGH, bbox_inches='tight')
        plt.close()

        print(f"相关系数热力图已保存至: {save_path}")

    def plot_standard_radar(self, save_path):
        """绘制标准响应模式雷达图（低多样性场景）"""
        print("\n正在生成标准响应模式雷达图...")

        # 计算中心模式
        center_pattern = np.mean(self.P15_matrix, axis=0)
        baseline = np.min(center_pattern)

        # 归一化
        normalized = (center_pattern - baseline) / (np.max(center_pattern) - baseline + 1e-6)

        categories = [Config.get_sensor_name(i, self.version_config) for i in range(self.num_sensors)]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        normalized = np.concatenate((normalized, [normalized[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=Config.VIS_FIGSIZE_RADAR, subplot_kw=dict(projection='polar'))

        ax.plot(angles, normalized, 'o-', linewidth=2.5,
                label='标准响应模式', color='blue')
        ax.fill(angles, normalized, alpha=Config.VIS_ALPHA_FILL, color='blue')

        # 如果有异常样本，叠加绘制
        if len(self.outlier_samples) > 0 and len(self.outlier_samples) <= Config.VIS_MAX_OUTLIER_RADAR:
            for outlier in self.outlier_samples[:Config.VIS_MAX_OUTLIER_RADAR]:
                idx = outlier['index']
                outlier_pattern = self.P15_matrix[idx]
                outlier_normalized = (outlier_pattern - baseline) / (np.max(center_pattern) - baseline + 1e-6)
                outlier_normalized = np.concatenate((outlier_normalized, [outlier_normalized[0]]))

                ax.plot(angles, outlier_normalized, 'o-', linewidth=1.5,
                        label=f"异常样本: {outlier['filename'][:20]}...", alpha=Config.VIS_ALPHA_OUTLIER)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)

        ax.set_title('标准响应模式雷达图\n（蓝色=标准模式，其他=异常样本）',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=Config.VIS_DPI, bbox_inches='tight')
        plt.close()

        print(f"标准雷达图已保存至: {save_path}")

    def plot_multi_gas_radar(self, save_path):
        """绘制多气体对比雷达图（高多样性场景）"""
        print("\n正在生成多气体对比雷达图...")

        # 按气体类型分组
        gas_patterns = {}
        for i, metadata in enumerate(self.file_metadata):
            key = f"{metadata['gas_type']}-{metadata['concentration']}"
            if key not in gas_patterns:
                gas_patterns[key] = []
            gas_patterns[key].append(self.P15_matrix[i])

        # 计算每种气体的中心模式
        gas_centers = {}
        for gas, patterns in gas_patterns.items():
            gas_centers[gas] = np.mean(patterns, axis=0)

        # 全局归一化
        all_values = np.concatenate(list(gas_centers.values()))
        global_min = np.min(all_values)
        global_max = np.max(all_values)

        categories = [Config.get_sensor_name(i, self.version_config) for i in range(self.num_sensors)]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=Config.VIS_FIGSIZE_RADAR_MULTI, subplot_kw=dict(projection='polar'))

        colors = plt.cm.Set3(np.linspace(0, 1, len(gas_centers)))

        for (gas, center), color in zip(gas_centers.items(), colors):
            normalized = (center - global_min) / (global_max - global_min + 1e-6)
            normalized = np.concatenate((normalized, [normalized[0]]))

            ax.plot(angles, normalized, 'o-', linewidth=2,
                    label=gas, color=color)
            ax.fill(angles, normalized, alpha=Config.VIS_ALPHA_FILL_MULTI, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)

        ax.set_title('多气体响应模式对比雷达图',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=Config.VIS_DPI, bbox_inches='tight')
        plt.close()

        print(f"多气体雷达图已保存至: {save_path}")

    def plot_stability_table(self, save_path):
        """绘制稳定性统计表（低多样性场景）"""
        print("\n正在生成稳定性统计表...")

        stability = self.results['stability']

        table_data = []
        for i in range(self.num_sensors):
            sensor_name = Config.get_sensor_name(i, self.version_config)
            table_data.append([
                sensor_name,
                f"{stability['means'][i]:.2f}",
                f"{stability['stds'][i]:.2f}",
                f"{stability['cvs'][i]:.3f}",
                "稳定" if stability['cvs'][i] < Config.STABILITY_CV_STABLE else
                ("一般" if stability['cvs'][i] < Config.STABILITY_CV_UNSTABLE else "不稳定")
            ])

        df = pd.DataFrame(table_data,
                          columns=['传感器', '均值', '标准差', '变异系数CV', '稳定性'])

        fig_height = max(Config.VIS_FIGSIZE_TABLE[1], self.num_sensors * 0.3)
        fig, ax = plt.subplots(figsize=(Config.VIS_FIGSIZE_TABLE[0], fig_height))

        # 为标题预留顶部空间
        plt.subplots_adjust(top=0.94, bottom=0.01, left=0.01, right=0.99)
        ax.axis('off')

        table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         cellLoc='center',
                         loc='upper center',
                         colWidths=[0.15, 0.15, 0.15, 0.2, 0.15])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # 设置表头样式
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 根据稳定性着色
        for i in range(len(df)):
            if df.iloc[i]['稳定性'] == '不稳定':
                for j in range(len(df.columns)):
                    table[(i + 1, j)].set_facecolor('#ffcccc')
            elif df.iloc[i]['稳定性'] == '稳定':
                for j in range(len(df.columns)):
                    table[(i + 1, j)].set_facecolor('#ccffcc')

        ax.set_title('传感器稳定性统计表', fontsize=14, fontweight='bold', pad=10)
        plt.savefig(save_path, dpi=Config.VIS_DPI, bbox_inches='tight')
        plt.close()

        print(f"稳定性统计表已保存至: {save_path}")

    def plot_discrimination_ranking(self, save_path):
        """绘制区分度排名图（高多样性场景）"""
        print("\n正在生成区分度排名图...")

        disc_df = self.results['discrimination']

        top20 = disc_df.head(20)

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['green' if score > Config.DISCRIMINATION_SCORE_STRONG
                  else 'orange' if score > Config.DISCRIMINATION_SCORE_NORMAL
                  else 'red'
                  for score in top20['Score']]

        bars = ax.barh(top20['Sensor'], top20['Score'], color=colors, alpha=Config.VIS_ALPHA_BOX)

        ax.set_xlabel('综合区分度评分', fontsize=12)
        ax.set_ylabel('传感器', fontsize=12)
        ax.set_title('传感器区分度排名（前20）\n绿色=优秀，橙色=良好，红色=一般',
                     fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.0])
        ax.grid(True, alpha=Config.VIS_ALPHA_GRID, axis='x')

        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, top20['Score'])):
            ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{score:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=Config.VIS_DPI, bbox_inches='tight')
        plt.close()

        print(f"区分度排名图已保存至: {save_path}")

    def run_full_analysis(self, save_dir='./multi_file_output'):
        """运行完整分析流程"""
        try:
            # 1. 加载数据
            self.load_and_extract_data()

            # 2. 硬件环境体检
            self.analyze_environment()

            # 3. 判断样本多样性
            self.check_diversity()

            # 4. 统计分析
            self.analyze_statistics()

            # 5. 异常检测
            self.detect_outliers()

            # 6. 相关性分析
            self.calculate_correlation()

            # 7. 生成报告
            os.makedirs(save_dir, exist_ok=True)
            report_path = os.path.join(save_dir, 'multi_file_report.txt')
            self.generate_report(report_path)

            # 7. 可视化
            self.visualize_all(save_dir)

            print(f"\n{'=' * 60}")
            print("[OK] 多文件EDA分析完成！")
            print(f"所有结果已保存至: {save_dir}")
            print(f"{'=' * 60}\n")

            return self

        except Exception as e:
            print(f"\n[ERROR] 分析过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# ====================
# 命令行入口
# ====================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python multi.py <input_folder_path>", file=sys.stderr)
        sys.exit(1)

    folder_path = sys.argv[1]

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"[ERROR] 文件夹不存在: {folder_path}", file=sys.stderr)
        sys.exit(1)

    # 加载文件夹中的所有txt文件
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))

    if len(file_paths) == 0:
        print(f"[ERROR] 文件夹中没有找到txt文件: {folder_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 找到 {len(file_paths)} 个txt文件")

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 输出目录
    output_dir = os.path.join(script_dir, "EDATestResult")

    # 清空并重建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"[INFO] 已清空旧的输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 输出目录: {output_dir}")

    # 创建多文件EDA对象
    multi_eda = MultiFileEDA(file_paths)

    # 运行完整分析
    result = multi_eda.run_full_analysis(save_dir=output_dir)

    if result:
        print("\n分析完成！")
        print(f"样本多样性: {result.sample_diversity}")
        print(f"异常样本数: {len(result.outlier_samples)}")
