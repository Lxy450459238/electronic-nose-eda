# 电子鼻单文件EDA分析 - 自动识别新旧版本
# 修改时间: 2026.06.08
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import iqr
import os
import sys
import shutil
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


class SingleFileEDA:
    """单文件探索性数据分析"""

    def __init__(self, file_path, skip_first_row=True):
        """
        初始化

        参数:
            file_path: str, txt文件路径
            skip_first_row: bool, 是否跳过首行（如果核心传感器全是0）
        """
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.skip_first_row = skip_first_row

        self.version_config = None
        self.num_sensors = None
        self.data = None
        self.env_data = None              # 环境与系统通道数据
        self.stages = {}
        self.metadata = {}
        self.quality_score = 0
        self.warnings = []
        self.results = {}
        self.responsive_sensors = None    # 有响应传感器索引
        self.non_responsive_sensors = None  # 无响应传感器索引

    def parse_file(self):
        """解析文件，识别阶段"""
        print(f"\n{'=' * 60}")
        print(f"正在分析文件: {self.filename}")
        print(f"{'=' * 60}")

        # 读取数据
        self.data = pd.read_csv(self.file_path, sep=' ', header=None,
                                encoding='utf-8', on_bad_lines='skip')

        # 自动检测版本
        data_columns = self.data.shape[1]
        print(f"\n[检测] 数据文件列数: {data_columns}")

        try:
            self.version_config = Config.detect_version(data_columns)
            self.num_sensors = self.version_config['sensor_count']
            print(f"[OK] 检测到{self.version_config['name']}")
            print(f"[OK] 传感器数量: {self.num_sensors}个\n")
        except ValueError as e:
            raise ValueError(str(e))

        # 提取元数据（从文件名）
        self._parse_metadata()

        # 识别阶段
        stage_column = self.data.iloc[:, 4]
        sensor_data = Config.extract_sensor_data(self.data, self.version_config)

        # 提取环境监控数据
        try:
            raw_env = Config.extract_env_data(self.data, self.version_config)
            if raw_env is not None:
                self.env_data = raw_env.astype(float)
        except Exception as e:
            print(f"  [WARNING] 环境数据提取失败: {e}")

        time_column = self.data.iloc[:, 2]

        for stage in ['P13', 'P14', 'P15', 'P16', 'P17', 'P18']:
            stage_mask = stage_column == stage
            if stage_mask.any():
                stage_data = sensor_data[stage_mask]

                time_array = time_column[stage_mask].values
                stage_mask_indices = np.where(stage_mask)[0]

                # 首行跳过逻辑与同步裁切
                if self.skip_first_row and len(stage_data) > Config.DATA_SKIP_ROWS_MAX:
                    valid_start = 0
                    for i in range(Config.DATA_SKIP_ROWS_MAX):
                        check_cols = min(Config.DATA_SKIP_CHECK_COLS, self.num_sensors)
                        if np.any(stage_data[i, :check_cols] == 0):
                            valid_start = i + 1
                        else:
                            break

                    if valid_start > 0:
                        print(f"  [INFO] {stage}前 {valid_start} 行核心传感器包含0值，已同步裁切")
                        stage_data = stage_data[valid_start:]
                        time_array = time_array[valid_start:]
                        stage_mask_indices = stage_mask_indices[valid_start:]

                self.stages[stage] = {
                    'data': stage_data,
                    'time': time_array,
                    'indices': stage_mask_indices
                }
                print(f"[OK] 检测到 {stage}: {len(stage_data)} 个数据点")

        if not self.stages:
            raise ValueError("未检测到任何有效阶段标签！")

        return self

    def _parse_metadata(self):
        """从文件名提取元数据 - 智能日期提取（正则匹配8位连续数字）"""
        name_no_ext = self.filename.replace('.txt', '')
        parts = name_no_ext.split('-')

        # 智能提取日期：寻找连续的8位以上数字 (如 20211013 或 20230320)
        date_match = re.search(r'20\d{6}', name_no_ext)
        date_str = date_match.group(0) if date_match else 'unknown'

        gas_type = 'unknown'
        concentration = 'unknown'

        if len(parts) >= 4:
            # 兼容新旧格式：如果第一段就是日期则按标准格式解析
            if parts[0] == date_str:
                gas_type = parts[2] if len(parts) > 2 else 'unknown'
                concentration = parts[3] if len(parts) > 3 else 'unknown'
            else:
                gas_type = parts[2] if len(parts) > 2 else 'unknown'
                concentration = parts[3] if len(parts) > 3 else 'unknown'

        self.metadata = {
            'date': date_str,
            'id': parts[1] if len(parts) > 1 else 'unknown',
            'gas_type': gas_type,
            'concentration': concentration,
            'replicate': parts[4] if len(parts) > 4 else '1'
        }

    def analyze_baseline(self):
        """分析基线阶段（P13或P14）- 四维融合判定规则"""
        baseline_stage = None

        if 'P13' in self.stages:
            P13_data = self.stages['P13']['data']

            # 计算P13阶段的基础指标
            P13_mean = np.mean(P13_data, axis=0)
            P13_std = np.std(P13_data, axis=0)
            P13_cv = P13_std / (np.abs(P13_mean) + 1e-6)
            P13_ptp = np.max(P13_data, axis=0) - np.min(P13_data, axis=0)

            # 四维判定规则 (P13初筛)
            cv_mask_13 = (np.abs(P13_mean) >= Config.BASELINE_MIN_MEAN_THRESHOLD) & (
                        P13_cv >= Config.BASELINE_CV_STABLE)
            small_std_mask_13 = (np.abs(P13_mean) < Config.BASELINE_MIN_MEAN_THRESHOLD) & (
                        P13_std >= Config.BASELINE_MAX_STD_FALLBACK)
            large_std_mask_13 = P13_std >= Config.BASELINE_MAX_ALLOWED_STD
            ptp_mask_13 = P13_ptp >= Config.BASELINE_MAX_PTP_DRIFT

            P13_unstable_mask = cv_mask_13 | small_std_mask_13 | large_std_mask_13 | ptp_mask_13
            unstable_count = np.sum(P13_unstable_mask)

            max_unstable_allowed = int(self.num_sensors * Config.BASELINE_UNSTABLE_RATIO)

            if unstable_count <= max_unstable_allowed:
                baseline_stage = 'P13'
                print(f"\n--- 基线阶段分析 (P13) [允许不稳定上限: {max_unstable_allowed}个] ---")
            else:
                print(f"\n--- P13基线不稳定（{unstable_count}个传感器未达标，超过阈值 {max_unstable_allowed}）---")
                if 'P14' in self.stages:
                    baseline_stage = 'P14'
                    print("--- 改用P14作为基线 ---")
                else:
                    baseline_stage = 'P13'
                    print("--- 未找到P14，继续使用P13（需注意）---")
                    self.warnings.append(f"[WARNING] P13基线不稳定，但无P14替代")

        elif 'P14' in self.stages:
            baseline_stage = 'P14'
            print("\n--- 基线阶段分析 (P14) ---")
        else:
            self.warnings.append("[WARNING] 未找到基线数据（P13或P14）")
            return self

        # 分析选定的基线
        baseline_data = self.stages[baseline_stage]['data']

        baseline_mean = np.mean(baseline_data, axis=0)
        baseline_std = np.std(baseline_data, axis=0)
        baseline_var = np.var(baseline_data, axis=0)
        baseline_cv = baseline_std / (np.abs(baseline_mean) + 1e-6)
        baseline_ptp = np.max(baseline_data, axis=0) - np.min(baseline_data, axis=0)

        self.results['baseline'] = {
            'stage': baseline_stage,
            'mean': baseline_mean,
            'std': baseline_std,
            'var': baseline_var,
            'cv': baseline_cv,
            'ptp': baseline_ptp
        }

        # 检查负值传感器
        negative_sensors = np.where(baseline_mean < 0)[0]
        if len(negative_sensors) > 0:
            print(f"  [INFO] {len(negative_sensors)}个传感器基线为负值（正常，可能是差分输出）")

        # 最终基线的四维判定规则
        cv_mask = (np.abs(baseline_mean) >= Config.BASELINE_MIN_MEAN_THRESHOLD) & (
                    baseline_cv >= Config.BASELINE_CV_STABLE)
        small_std_mask = (np.abs(baseline_mean) < Config.BASELINE_MIN_MEAN_THRESHOLD) & (
                    baseline_std >= Config.BASELINE_MAX_STD_FALLBACK)
        large_std_mask = baseline_std >= Config.BASELINE_MAX_ALLOWED_STD
        ptp_mask = baseline_ptp >= Config.BASELINE_MAX_PTP_DRIFT

        unstable_mask = cv_mask | small_std_mask | large_std_mask | ptp_mask
        unstable_sensors = np.where(unstable_mask)[0]
        max_unstable_allowed = int(self.num_sensors * Config.BASELINE_UNSTABLE_RATIO)

        if len(unstable_sensors) > max_unstable_allowed:
            self.warnings.append(f"[WARNING] {baseline_stage}不稳定：{len(unstable_sensors)}个传感器未达标")
            self.quality_score -= 15
            print(f"  [WARNING] 不稳定传感器总体超标 (大于上限 {max_unstable_allowed} 个)")
        else:
            print(f"  [OK] {baseline_stage}稳定性良好 (不稳定数量 {len(unstable_sensors)} <= {max_unstable_allowed})")
            self.quality_score += 20

        if len(unstable_sensors) > 0:
            print(f"  [INFO] 具体不稳定通道明细 ({len(unstable_sensors)}个):")
            for idx in unstable_sensors:
                real_name = Config.get_sensor_name(idx, self.version_config)
                if ptp_mask[idx]:
                    print(
                        f"     {idx}({real_name}): 极差落差={baseline_ptp[idx]:.1f} >= {Config.BASELINE_MAX_PTP_DRIFT} (触发基线波动幅度过大规则)")
                elif large_std_mask[idx]:
                    print(
                        f"     {idx}({real_name}): STD={baseline_std[idx]:.1f} >= {Config.BASELINE_MAX_ALLOWED_STD} (触发全局大波动规则)")
                elif cv_mask[idx]:
                    print(
                        f"     {idx}({real_name}): CV={baseline_cv[idx]:.3f} >= {Config.BASELINE_CV_STABLE} (均值={baseline_mean[idx]:.1f})")
                else:
                    print(
                        f"     {idx}({real_name}): STD={baseline_std[idx]:.3f} >= {Config.BASELINE_MAX_STD_FALLBACK} (触发小均值规则)")

        # 异常点（毛刺）精细化占比统计
        Q1 = np.percentile(baseline_data, 25, axis=0)
        Q3 = np.percentile(baseline_data, 75, axis=0)
        IQR = Q3 - Q1

        # 针对极小均值/零均值通道的 IQR 保护
        IQR = np.where(IQR == 0, 1e-5, IQR)

        lower_bound = Q1 - Config.BASELINE_IQR_MULTIPLIER * IQR
        upper_bound = Q3 + Config.BASELINE_IQR_MULTIPLIER * IQR

        outliers_mask = (baseline_data < lower_bound) | (baseline_data > upper_bound)

        outliers_per_sensor = np.sum(outliers_mask, axis=0)
        total_samples = baseline_data.shape[0]
        outlier_ratios = outliers_per_sensor / total_samples

        bad_outlier_sensors = []
        for i in range(self.num_sensors):
            if outlier_ratios[i] > Config.BASELINE_OUTLIER_RATIO_THRESHOLD:
                sensor_name = Config.get_sensor_name(i, self.version_config)
                bad_outlier_sensors.append((sensor_name, outlier_ratios[i], outliers_per_sensor[i]))

        self.results['baseline']['outliers_per_sensor'] = outliers_per_sensor
        self.results['baseline']['bad_outlier_sensors'] = bad_outlier_sensors

        if bad_outlier_sensors:
            threshold_pct = Config.BASELINE_OUTLIER_RATIO_THRESHOLD * 100
            self.warnings.append(
                f"[WARNING] {baseline_stage}存在 {len(bad_outlier_sensors)} 个传感器毛刺超标 (>{threshold_pct:.1f}%)")
            print(f"  [WARNING] 发现高频毛刺干扰通道 (异常比例 > {threshold_pct:.1f}%):")
            for name, ratio, count in bad_outlier_sensors:
                print(f"     {name}: 异常率 {ratio * 100:.1f}% ({count}/{total_samples}个异常点)")
        else:
            print(f"  [OK] {baseline_stage}所有通道均无严重高频毛刺干扰。")

        return self

    def extract_steady_state(self, stage_data, stage_name):
        """提取稳态数据"""
        steady_values = []
        no_steady_count = 0

        for sensor_idx in range(self.num_sensors):
            sensor_series = stage_data[:, sensor_idx]

            if len(sensor_series) < 30:
                steady_values.append(np.mean(sensor_series))
                continue

            window_size = min(10, len(sensor_series) // 4)
            stable_segment = None

            for i in range(len(sensor_series) - window_size - 20):
                window = sensor_series[i:i + window_size]
                window_mean = np.mean(window)

                if window_mean == 0:
                    continue

                if np.std(window) < abs(window_mean) * 0.05:
                    next_segment = sensor_series[i + window_size:min(i + window_size + 20, len(sensor_series))]
                    if len(next_segment) > 0:
                        diffs = np.abs(np.diff(next_segment))
                        if np.all(diffs < np.abs(np.mean(next_segment)) * 0.05):
                            stable_segment = sensor_series[i:i + window_size + len(next_segment)]
                            break

            if stable_segment is None:
                stable_segment = sensor_series[len(sensor_series) // 2:]
                no_steady_count += 1

            steady_values.append(np.mean(stable_segment))

        if no_steady_count > 10:
            self.warnings.append(f"[WARNING] {stage_name}: {no_steady_count}个传感器未检测到明确稳态段")

        return np.array(steady_values)

    def analyze_sampling(self):
        """分析进样阶段（动态支持 P15 或 P17）"""
        print("\n--- 进样阶段分析 ---")

        sample_stage = 'P15' if 'P15' in self.stages else 'P17' if 'P17' in self.stages else None

        if not sample_stage:
            self.warnings.append("[WARNING] 未找到进样阶段(P15或P17)数据")
            print("  [WARNING] 未找到进样阶段数据，跳过进样分析")
            return self

        print(f"\n  [{sample_stage} 分析]")
        sample_data = self.stages[sample_stage]['data']
        sample_steady = self.extract_steady_state(sample_data, sample_stage)

        self.results['sample_stage'] = sample_stage
        self.results['sample_steady'] = sample_steady

        if 'baseline' in self.results:
            baseline = self.results['baseline']['mean']
            response_delta = sample_steady - baseline
            response_ratio = np.abs(response_delta) / (np.abs(baseline) + 1e-6)

            responsive_sensors = np.sum(response_ratio > 0.1)
            print(f"    有效响应传感器: {responsive_sensors}/{self.num_sensors}")

            self.responsive_sensors = np.where(response_ratio > 0.1)[0]
            self.non_responsive_sensors = np.where(response_ratio <= 0.1)[0]

            responsive_details = [f"{idx}({Config.get_sensor_name(idx, self.version_config)})" for idx in
                                  self.responsive_sensors]
            print(f"    响应传感器明细: {responsive_details}")
            print(f"    不响应传感器数: {len(self.non_responsive_sensors)}")

            if responsive_sensors < 10:
                self.warnings.append(f"[WARNING] {sample_stage}有效响应传感器过少")
                self.quality_score -= 15
            else:
                self.quality_score += 25

            self.results['sample_response'] = {
                'delta': response_delta,
                'ratio': response_ratio,
                'responsive_count': responsive_sensors
            }

        # 如果 P15 和 P17 同时存在，进行一致性比对
        if 'P15' in self.stages and 'P17' in self.stages and sample_stage == 'P15':
            print("\n  [P17 对比分析]")
            P17_data = self.stages['P17']['data']
            P17_steady = self.extract_steady_state(P17_data, 'P17')
            correlation = np.corrcoef(sample_steady, P17_steady)[0, 1]
            print(f"    P15与P17相关性: {correlation:.3f}")

            if correlation > 0.95:
                print(f"    [OK] 两次进样高度一致")
                self.quality_score += 10
            elif correlation < 0.7:
                self.warnings.append(f"[WARNING] P15与P17差异较大 (r={correlation:.3f})")
                self.quality_score -= 10

        return self

    def calculate_temporal_correlation(self):
        """计算时间序列相关系数矩阵（动态阶段）"""
        print("\n--- 时间序列相关性分析 ---")

        sample_stage = self.results.get('sample_stage')
        if not sample_stage or sample_stage not in self.stages:
            print("  未找到进样阶段数据，跳过时间序列相关性分析")
            return self

        sample_data = self.stages[sample_stage]['data']
        dynamic_length = len(sample_data) // 2
        dynamic_data = sample_data[:dynamic_length, :]

        print(f"  使用{sample_stage}前{dynamic_length}个时间点计算时间序列相关性")

        corr_matrix = np.corrcoef(dynamic_data.T)

        high_corr_pairs = []
        for i in range(self.num_sensors):
            for j in range(i + 1, self.num_sensors):
                if abs(corr_matrix[i, j]) > 0.8:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))

        print(f"  发现{len(high_corr_pairs)}对高度相关的传感器（|r|>0.8）")

        self.results['temporal_correlation'] = {
            'matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs
        }

        return self

    def extract_cleaning_steady_state(self, stage_data, baseline):
        """
        提取清洗后稳态值（从末端向前搜索）

        参数:
            stage_data: (n, num_sensors) 清洗阶段数据
            baseline: (num_sensors,) 基线均值

        返回:
            steady_values: (num_sensors,) 清洗后稳态值
        """
        steady_values = []
        methods_count = {'steady': 0, 'tail40': 0, 'short': 0}

        for sensor_idx in range(self.num_sensors):
            sensor_series = stage_data[:, sensor_idx]
            baseline_sensor = baseline[sensor_idx]

            if len(sensor_series) < 20:
                steady_values.append(np.mean(sensor_series[len(sensor_series) // 2:]))
                methods_count['short'] += 1
                continue

            steady_segment = None
            window_size = min(10, len(sensor_series) // 5)

            # 从末端向前搜索稳态段
            for i in range(len(sensor_series) - window_size, max(0, len(sensor_series) // 2), -1):
                window = sensor_series[i:i + window_size]

                # 稳态条件：窗口波动 < 基线绝对值的5%
                if np.std(window) < abs(baseline_sensor) * 0.05 + 1e-6:
                    steady_segment = window
                    methods_count['steady'] += 1
                    break

            # 如果没找到稳态，使用后40%均值
            if steady_segment is None:
                cutoff = int(len(sensor_series) * 0.6)
                steady_segment = sensor_series[cutoff:]
                methods_count['tail40'] += 1

            steady_values.append(np.mean(steady_segment))

        if methods_count['tail40'] > 10:
            print(f"  [INFO] 稳态识别: {methods_count['steady']}个传感器, "
                  f"后40%均值: {methods_count['tail40']}个传感器")

        return np.array(steady_values)

    def analyze_cleaning(self):
        """分析清洗阶段 - 动态匹配 (P15->P16 或 P17->P18)"""
        if 'baseline' not in self.results:
            return self

        sample_stage = self.results.get('sample_stage')
        if not sample_stage or 'sample_steady' not in self.results:
            print("\n--- 清洗阶段分析 ---")
            print("  [WARNING] 缺少进样稳态数据，无法计算归一化清洗度")
            return self

        # 动态推断对应的清洗阶段（兼容旧版P15->P18逻辑）
        if sample_stage == 'P15':
            clean_stage = 'P16' if 'P16' in self.stages else 'P18' if 'P18' in self.stages else None
        elif sample_stage == 'P17':
            clean_stage = 'P18' if 'P18' in self.stages else None
        else:
            clean_stage = None

        if not clean_stage or clean_stage not in self.stages:
            print("\n--- 清洗阶段分析 ---")
            print(f"  [WARNING] 未找到对应的清洗阶段({clean_stage})数据")
            return self

        print(f"\n--- 清洗阶段分析 ({clean_stage}) ---")

        baseline = self.results['baseline']['mean']
        sample_steady = self.results['sample_steady']
        clean_data = self.stages[clean_stage]['data']
        clean_steady = self.extract_cleaning_steady_state(clean_data, baseline)

        response_amplitude = sample_steady - baseline
        residual_delta = clean_steady - baseline

        if self.responsive_sensors is not None and len(self.responsive_sensors) > 0:
            eval_sensors = self.responsive_sensors
            print(f"  [INFO] 仅评估 {len(eval_sensors)} 个响应传感器（排除不响应传感器）")
        else:
            eval_sensors = list(range(self.num_sensors))
            print("  [INFO] 评估所有传感器")

        print("\n  === 清洗诊断（前3个响应传感器）===")
        for i, sensor_idx in enumerate(eval_sensors[:3]):
            resp_amp = response_amplitude[sensor_idx]
            resid = residual_delta[sensor_idx]

            real_name = Config.get_sensor_name(sensor_idx, self.version_config)
            print(f"  {sensor_idx} ({real_name}):")
            print(f"    基线: {baseline[sensor_idx]:.2f}")
            print(f"    {sample_stage}进样稳态: {sample_steady[sensor_idx]:.2f}")
            print(f"    {clean_stage}清洗稳态: {clean_steady[sensor_idx]:.2f}")
            print(f"    响应幅度: {resp_amp:.2f}")
            print(f"    清洗后残留: {resid:.2f}")
            if resp_amp * resid < 0:
                print(f"    [WARNING] 已越过基线（符号相反）")
            elif abs(resp_amp) > 1e-6:
                print(f"    归一化残留率: {abs(resid) / abs(resp_amp):.3f}")

        recovery_rate = np.zeros(self.num_sensors)
        over_recovery_count = 0
        drift_warning = []

        for sensor_idx in eval_sensors:
            resp_amp = response_amplitude[sensor_idx]
            resid = residual_delta[sensor_idx]

            if abs(resp_amp) < 1e-6:
                recovery_rate[sensor_idx] = 0
                continue

            if resp_amp * resid < 0:
                recovery_rate[sensor_idx] = 0
                over_recovery_count += 1
                if abs(resid) > abs(resp_amp) * 0.5:
                    drift_warning.append(sensor_idx)
            else:
                recovery_rate[sensor_idx] = abs(resid) / abs(resp_amp)

        if over_recovery_count > 0:
            print(f"\n  [INFO] {over_recovery_count}个传感器已越过基线")
            if len(drift_warning) > 0:
                drift_names = [Config.get_sensor_name(int(idx), self.version_config) for idx in drift_warning]
                print(f"  [WARNING] 其中{len(drift_warning)}个传感器可能存在漂移或过度清洗: {drift_names[:5]}")

                # 警告信息带上具体传感器名字
                drift_str = ", ".join(drift_names[:5])
                if len(drift_names) > 5:
                    drift_str += " 等"
                self.warnings.append(f"[WARNING] {clean_stage}阶段存在传感器漂移或过度清洗: {drift_str}")

        print(f"\n  残留率统计 (归一化距离，已处理越界):")
        print(f"    最小: {recovery_rate[eval_sensors].min():.3f}")
        print(f"    最大: {recovery_rate[eval_sensors].max():.3f}")
        print(f"    平均: {recovery_rate[eval_sensors].mean():.3f}")
        print(f"    中位数: {np.median(recovery_rate[eval_sensors]):.3f}")

        avg_recovery = np.mean(recovery_rate[eval_sensors])
        cleaning_degree = (1 - avg_recovery) * 100  # 允许负值，负值表示漂移（残留超过响应幅度）

        print(f"\n  {clean_stage}平均清洗度: {cleaning_degree:.1f}%")
        if avg_recovery > 1.0:
            print(f"    [INFO] 平均残留率超过100%，传感器在清洗阶段可能发生漂移而非单纯未恢复")

        eval_recovery = recovery_rate[eval_sensors]
        excellent = np.sum(eval_recovery < 0.05)
        good = np.sum((eval_recovery >= 0.05) & (eval_recovery < 0.15))
        poor = np.sum(eval_recovery >= 0.25)
        threshold = int(len(eval_sensors) * 0.8)

        if excellent + good >= threshold:
            print(f"    [OK] 清洗充分 (优秀:{excellent}, 良好:{good})")
            self.quality_score += 10
        elif avg_recovery <= 0.15:
            print(f"    [OK] 清洗可接受")
            self.quality_score += 5
        else:
            print(f"    [WARNING] 清洗不充分")
            self.warnings.append(f"[WARNING] {clean_stage}清洗不充分 (平均残留{avg_recovery * 100:.1f}%)")
            self.quality_score -= 10

        if poor > 0 and poor < len(eval_sensors):
            poor_mask = recovery_rate >= 0.25
            poor_responsive = [idx for idx in eval_sensors if poor_mask[idx]]
            if len(poor_responsive) > 0:
                print(f"    [WARNING] {len(poor_responsive)}个响应传感器恢复不足:")
                for idx in poor_responsive[:3]:
                    real_name = Config.get_sensor_name(idx, self.version_config)
                    print(f"      {idx} ({real_name}): 残留{recovery_rate[idx] * 100:.1f}%")

        # 统一保存清洗结果
        self.results['clean_stage'] = clean_stage
        self.results['clean_recovery'] = avg_recovery
        self.results['clean_steady'] = clean_steady
        self.results['cleaning_degree'] = cleaning_degree

        return self

    def analyze_environment(self):
        """全局环境与系统通道体检"""
        print("\n--- 全局硬件健康体检 ---")
        if self.env_data is None:
            print("  [INFO] 未配置环境通道，跳过体检。")
            return self

        # 计算整个实验周期的统计量
        env_mean = np.nanmean(self.env_data, axis=0)
        env_std = np.nanstd(self.env_data, axis=0)
        env_max = np.nanmax(self.env_data, axis=0)
        env_min = np.nanmin(self.env_data, axis=0)

        abnormal_env = []
        env_name_map = self.version_config.get('env_name_map', {})

        for i in range(self.env_data.shape[1]):
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
                abnormal_env.append((name, reason))

        self.results['environment'] = abnormal_env

        if abnormal_env:
            self.warnings.append(f"[WARNING] 硬件体检异常: {len(abnormal_env)} 项环境指标未达标")
            for name, reason in abnormal_env:
                print(f"    [!] {name}: {reason}")
        else:
            print("  [OK] 所有供电、温度、湿度、气压在整个实验全周期内均保持稳定。")

        return self

    def calculate_quality_score(self):
        """计算综合质量评分"""
        self.quality_score += 50
        self.quality_score = max(0, min(100, self.quality_score))
        return self

    def generate_report(self):
        """生成文本报告"""
        print(f"\n{'=' * 60}")
        print("质量评估报告")
        print(f"{'=' * 60}")

        print(f"\n文件名: {self.filename}")
        print(f"气体类型: {self.metadata.get('gas_type', 'unknown')}")
        print(f"浓度: {self.metadata.get('concentration', 'unknown')}")
        print(f"日期: {self.metadata.get('date', 'unknown')}")

        print(f"\n阶段识别结果:")
        for stage in self.stages.keys():
            print(f"  [OK] {stage}: {len(self.stages[stage]['data'])}个数据点")

        # 基线稳定性详情
        if 'baseline' in self.results:
            baseline_cv = self.results['baseline']['cv']
            baseline_mean = self.results['baseline']['mean']
            baseline_std = self.results['baseline']['std']
            baseline_ptp = self.results['baseline'].get('ptp', np.zeros_like(baseline_mean))
            baseline_stage = self.results['baseline']['stage']

            cv_stable_count = np.sum(baseline_cv < Config.BASELINE_CV_STABLE)

            # 综合四判定统计
            cv_mask = (np.abs(baseline_mean) >= Config.BASELINE_MIN_MEAN_THRESHOLD) & (
                        baseline_cv >= Config.BASELINE_CV_STABLE)
            small_std_mask = (np.abs(baseline_mean) < Config.BASELINE_MIN_MEAN_THRESHOLD) & (
                        baseline_std >= Config.BASELINE_MAX_STD_FALLBACK)
            large_std_mask = baseline_std >= Config.BASELINE_MAX_ALLOWED_STD
            ptp_mask = baseline_ptp >= Config.BASELINE_MAX_PTP_DRIFT

            unstable_mask = cv_mask | small_std_mask | large_std_mask | ptp_mask
            quad_stable_count = self.num_sensors - np.sum(unstable_mask)

            print(f"\n基线稳定性（{baseline_stage}）:")
            print(f"  稳定传感器（仅看CV<{Config.BASELINE_CV_STABLE * 100:.0f}%）: {cv_stable_count}/{self.num_sensors}")
            print(f"  稳定传感器（综合四判定）: {quad_stable_count}/{self.num_sensors}")
            print(f"  传感器CV范围: {baseline_cv.min():.3f} ~ {baseline_cv.max():.3f}")

            unstable = np.where(unstable_mask)[0]
            if len(unstable) > 0:
                print(f"  不稳定传感器明细:")
                for idx in unstable:
                    real_name = Config.get_sensor_name(idx, self.version_config)
                    if ptp_mask[idx]:
                        print(
                            f"    {real_name}: 极差落差={baseline_ptp[idx]:.1f} >= {Config.BASELINE_MAX_PTP_DRIFT} (基线波动幅度过大)")
                    elif large_std_mask[idx]:
                        print(
                            f"    {real_name}: STD={baseline_std[idx]:.1f} >= {Config.BASELINE_MAX_ALLOWED_STD} (全局大波动)")
                    elif cv_mask[idx]:
                        print(f"    {real_name}: CV={baseline_cv[idx]:.3f} >= {Config.BASELINE_CV_STABLE}")
                    else:
                        print(
                            f"    {real_name}: STD={baseline_std[idx]:.3f} >= {Config.BASELINE_MAX_STD_FALLBACK} (小均值波动超标)")

        if self.warnings:
            print(f"\n[WARNING] 警告信息 ({len(self.warnings)}个):")
            for warning in self.warnings:
                print(f"  {warning}")
        else:
            print(f"\n[OK] 未发现明显异常")

        print(f"\n综合质量评分: {self.quality_score}/100")

        if self.quality_score >= 80:
            recommendation = "优秀 - 建议保留用于后续分析"
        elif self.quality_score >= 60:
            recommendation = "良好 - 可以使用，但需注意警告信息"
        else:
            recommendation = "较差 - 建议人工复查或剔除"

        print(f"建议: {recommendation}")
        print(f"{'=' * 60}\n")

        return self

    def save_text_report(self, save_path):
        """保存文本报告到文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write("质量评估报告\n")
            f.write(f"{'=' * 60}\n\n")

            f.write(f"文件名: {self.filename}\n")
            f.write(f"气体类型: {self.metadata.get('gas_type', 'unknown')}\n")
            f.write(f"浓度: {self.metadata.get('concentration', 'unknown')}\n")
            f.write(f"日期: {self.metadata.get('date', 'unknown')}\n\n")

            f.write(f"阶段识别结果:\n")
            for stage in self.stages.keys():
                f.write(f"  [OK] {stage}: {len(self.stages[stage]['data'])}个数据点\n")

            # 写入硬件体检结果
            if 'environment' in self.results:
                f.write(f"\n硬件体检 (环境与系统通道):\n")
                abnormal_env = self.results['environment']
                if abnormal_env:
                    for name, reason in abnormal_env:
                        f.write(f"  [异常] {name}: {reason}\n")
                else:
                    f.write(f"  [OK] 实验全周期内供电、温度、湿度及气压均保持稳定。\n")

            # 基线稳定性详情
            if 'baseline' in self.results:
                baseline_cv = self.results['baseline']['cv']
                baseline_mean = self.results['baseline']['mean']
                baseline_std = self.results['baseline']['std']
                baseline_ptp = self.results['baseline'].get('ptp', np.zeros_like(baseline_mean))
                baseline_stage = self.results['baseline']['stage']

                cv_stable_count = np.sum(baseline_cv < Config.BASELINE_CV_STABLE)

                # 综合四判定统计
                cv_mask = (np.abs(baseline_mean) >= Config.BASELINE_MIN_MEAN_THRESHOLD) & (
                            baseline_cv >= Config.BASELINE_CV_STABLE)
                small_std_mask = (np.abs(baseline_mean) < Config.BASELINE_MIN_MEAN_THRESHOLD) & (
                            baseline_std >= Config.BASELINE_MAX_STD_FALLBACK)
                large_std_mask = baseline_std >= Config.BASELINE_MAX_ALLOWED_STD
                ptp_mask = baseline_ptp >= Config.BASELINE_MAX_PTP_DRIFT

                unstable_mask = cv_mask | small_std_mask | large_std_mask | ptp_mask
                quad_stable_count = self.num_sensors - np.sum(unstable_mask)

                f.write(f"\n基线稳定性（{baseline_stage}）:\n")
                f.write(
                    f"  稳定传感器（仅看CV<{Config.BASELINE_CV_STABLE * 100:.0f}%）: {cv_stable_count}/{self.num_sensors}\n")
                f.write(f"  稳定传感器（综合四判定）: {quad_stable_count}/{self.num_sensors}\n")
                f.write(f"  传感器CV范围: {baseline_cv.min():.3f} ~ {baseline_cv.max():.3f}\n")

                unstable = np.where(unstable_mask)[0]
                if len(unstable) > 0:
                    f.write(f"  不稳定传感器:\n")
                    for idx in unstable:
                        real_name = Config.get_sensor_name(idx, self.version_config)
                        if ptp_mask[idx]:
                            f.write(
                                f"    {real_name}: 极差落差={baseline_ptp[idx]:.1f} >= {Config.BASELINE_MAX_PTP_DRIFT} (基线波动幅度过大)\n")
                        elif large_std_mask[idx]:
                            f.write(
                                f"    {real_name}: STD={baseline_std[idx]:.1f} >= {Config.BASELINE_MAX_ALLOWED_STD} (全局大波动)\n")
                        elif cv_mask[idx]:
                            f.write(f"    {real_name}: CV={baseline_cv[idx]:.3f} >= {Config.BASELINE_CV_STABLE}\n")
                        else:
                            f.write(
                                f"    {real_name}: STD={baseline_std[idx]:.3f} >= {Config.BASELINE_MAX_STD_FALLBACK} (小均值通道)\n")

                # 写入异常点（毛刺）报告
                bad_outlier_sensors = self.results['baseline'].get('bad_outlier_sensors', [])
                if bad_outlier_sensors:
                    threshold_pct = Config.BASELINE_OUTLIER_RATIO_THRESHOLD * 100
                    f.write(f"\n  [持续性毛刺干扰检测] (阈值 > {threshold_pct:.1f}%):\n")
                    for name, ratio, count in bad_outlier_sensors:
                        f.write(f"    {name}: 异常率 {ratio * 100:.1f}% (含 {count} 个异常离群点)\n")
                else:
                    f.write(f"\n  [持续性毛刺干扰检测]: 未发现异常率超过阈值的传感器通道。\n")

            # 清洗度评估
            if 'cleaning_degree' in self.results:
                clean_stg = self.results.get('clean_stage', '清洗')
                cd = self.results['cleaning_degree']
                f.write(f"\n清洗度评估（{clean_stg}）:\n")
                f.write(f"  平均清洗度: {cd:.1f}%\n")
                if cd < 0:
                    f.write(f"  [INFO] 清洗度为负值，可能存在过度清洗或漂移（残留超过响应幅度）\n")
                elif cd >= 80:
                    f.write(f"  评价: 清洗充分\n")
                elif cd >= 60:
                    f.write(f"  评价: 清洗可接受\n")
                else:
                    f.write(f"  评价: 清洗不足，存在记忆效应风险\n")

            if self.warnings:
                f.write(f"\n[WARNING] 警告信息 ({len(self.warnings)}个):\n")
                for warning in self.warnings:
                    f.write(f"  {warning}\n")
            else:
                f.write(f"\n[OK] 未发现明显异常\n")

            f.write(f"\n综合质量评分: {self.quality_score}/100\n")

            if self.quality_score >= 80:
                recommendation = "优秀 - 建议保留用于后续分析"
            elif self.quality_score >= 60:
                recommendation = "良好 - 可以使用，但需注意警告信息"
            else:
                recommendation = "较差 - 建议人工复查或剔除"

            f.write(f"建议: {recommendation}\n")
            f.write(f"{'=' * 60}\n")

        print(f"文本报告已保存至: {save_path}")

    def plot_full_timeseries(self, save_path=None):
        """绘制完整时间序列（PNG）"""
        print("\n正在生成完整时间序列图...")

        full_data = Config.extract_sensor_data(self.data, self.version_config)
        stage_labels = self.data.iloc[:, 4].values
        time_points = np.arange(len(full_data))

        # 根据实际传感器数量生成足够的颜色
        if self.num_sensors <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, self.num_sensors))
        elif self.num_sensors <= 40:
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.tab20b(np.linspace(0, 1, self.num_sensors - 20))
            colors = np.vstack([colors1, colors2])
        else:
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.tab20b(np.linspace(0, 1, 20))
            remaining = self.num_sensors - 40
            colors3 = plt.cm.tab20c(np.linspace(0, 1, remaining))
            colors = np.vstack([colors1, colors2, colors3])

        fig, ax = plt.subplots(figsize=(20, 10))

        for i in range(self.num_sensors):
            real_name = Config.get_sensor_name(i, self.version_config)
            ax.plot(time_points, full_data[:, i],
                    label=real_name,
                    color=colors[i],
                    alpha=0.7,
                    linewidth=1.2)

        # 标注阶段边界
        stage_changes = [0]
        current_stage = stage_labels[0]

        for i in range(1, len(stage_labels)):
            if stage_labels[i] != current_stage:
                stage_changes.append(i)
                current_stage = stage_labels[i]
        stage_changes.append(len(stage_labels))

        stage_colors = {'P13': '#e8f4f8', 'P14': '#fff4e6', 'P15': '#f0f8e8',
                        'P16': '#fef0f0', 'P17': '#f0f8e8', 'P18': '#fef0f0'}

        for i in range(len(stage_changes) - 1):
            start_idx = stage_changes[i]
            end_idx = stage_changes[i + 1]
            stage = stage_labels[start_idx]

            ax.axvspan(start_idx, end_idx, alpha=0.2,
                       color=stage_colors.get(stage, 'gray'))

            if i > 0:
                ax.axvline(x=start_idx, color='red', linestyle='--',
                           linewidth=2, alpha=0.5)

            mid_point = (start_idx + end_idx) / 2
            ax.text(mid_point, ax.get_ylim()[1] * 0.95, stage,
                    ha='center', va='top', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('时间点 (采样点)', fontsize=14)
        ax.set_ylabel('传感器响应值', fontsize=14)
        ax.set_title(f'完整时间序列 - {self.filename}\n'
                     f'气体: {self.metadata.get("gas_type", "unknown")}, '
                     f'浓度: {self.metadata.get("concentration", "unknown")}',
                     fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  ncol=2, fontsize=8, framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"完整时间序列图已保存至: {save_path}")

        plt.close()

        return self

    def plot_baseline_boxplot(self, save_path=None):
        """绘制基线箱线图（PNG）"""
        if 'baseline' not in self.results:
            return self

        print("\n正在生成基线箱线图...")

        baseline_stage = self.results['baseline']['stage']
        baseline_data = self.stages[baseline_stage]['data']

        fig, ax = plt.subplots(figsize=(18, 6))

        bp = ax.boxplot(baseline_data, patch_artist=True,
                        showmeans=True, meanline=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax.set_title(f'{baseline_stage} 基线稳定性箱线图', fontsize=16, fontweight='bold')
        ax.set_xlabel('传感器编号', fontsize=14)
        ax.set_ylabel('响应值', fontsize=14)
        ax.set_xticks(range(1, self.num_sensors + 1))
        sensor_names = [Config.get_sensor_name(i, self.version_config) for i in range(self.num_sensors)]
        ax.set_xticklabels(sensor_names, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"基线箱线图已保存至: {save_path}")

        plt.close()

        return self

    def plot_response_delta(self, save_path=None):
        """绘制响应增量图（PNG）"""
        if 'sample_response' not in self.results:
            return self

        print("\n正在生成响应增量图...")

        response_delta = self.results['sample_response']['delta']
        sample_stage = self.results.get('sample_stage', '进样')

        colors_bar = ['green' if x > 0 else 'red' for x in response_delta]

        fig, ax = plt.subplots(figsize=(15, 6))

        ax.bar(range(self.num_sensors), response_delta, color=colors_bar, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

        ax.set_title(f'{sample_stage} 响应增量（相对基线）', fontsize=16, fontweight='bold')
        ax.set_xlabel('传感器编号', fontsize=14)
        ax.set_ylabel('响应增量', fontsize=14)
        ax.set_xticks(range(self.num_sensors))
        sensor_names = [Config.get_sensor_name(i, self.version_config) for i in range(self.num_sensors)]
        ax.set_xticklabels(sensor_names, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"响应增量图已保存至: {save_path}")

        plt.close()

        return self

    def plot_temporal_correlation_heatmap(self, save_path=None):
        """绘制时间序列相关系数热力图（PNG）"""
        if 'temporal_correlation' not in self.results:
            return self

        print("\n正在生成时间序列相关系数热力图...")

        corr_matrix = self.results['temporal_correlation']['matrix']
        sample_stage = self.results.get('sample_stage', '动态')

        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('相关系数', fontsize=12)

        ax.set_xticks(np.arange(self.num_sensors))
        ax.set_yticks(np.arange(self.num_sensors))

        sensor_names = [Config.get_sensor_name(i, self.version_config) for i in range(self.num_sensors)]
        ax.set_xticklabels(sensor_names, fontsize=8, rotation=90, ha='right')
        ax.set_yticklabels(sensor_names, fontsize=8)
        ax.set_title(f'时间序列相关系数矩阵（{sample_stage}阶段）\n反映传感器间的时间协同性',
                     fontsize=14, fontweight='bold', pad=20)

        ax.set_xlabel('传感器', fontsize=12)
        ax.set_ylabel('传感器', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"时间序列相关系数热力图已保存至: {save_path}")

        plt.close()

        return self

    def plot_radar_comparison(self, save_path=None):
        """绘制雷达图对比（基线->进样->清洗）"""
        print("\n正在生成雷达图对比...")

        if 'baseline' not in self.results or 'sample_steady' not in self.results:
            print("  缺少必要数据，跳过雷达图")
            return self

        baseline_mean = self.results['baseline']['mean']
        sample_steady = self.results['sample_steady']
        sample_stage = self.results['sample_stage']

        P15_normalized = (sample_steady - baseline_mean) / (np.abs(baseline_mean) + 1e-6)

        cleaning_normalized = None
        cleaning_label = None

        # 安全读取真实分析过的清洗阶段数据
        if 'clean_steady' in self.results and 'clean_stage' in self.results:
            clean_steady = self.results['clean_steady']
            clean_stage = self.results['clean_stage']
            cleaning_normalized = (clean_steady - baseline_mean) / (np.abs(baseline_mean) + 1e-6)
            cleaning_label = f'{clean_stage}（清洗后）'

        categories = [Config.get_sensor_name(i, self.version_config) for i in range(self.num_sensors)]
        N = len(categories)

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        P15_normalized = np.concatenate((P15_normalized, [P15_normalized[0]]))
        baseline_normalized = np.zeros(N + 1)

        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

        ax.plot(angles, baseline_normalized, 'o-', linewidth=2,
                label='基线（参考）', color='gray', linestyle='--', alpha=0.5)
        ax.fill(angles, baseline_normalized, alpha=0.1, color='gray')

        ax.plot(angles, P15_normalized, 'o-', linewidth=2,
                label=f'{sample_stage}（进样稳态）', color='blue')
        ax.fill(angles, P15_normalized, alpha=0.2, color='blue')

        if cleaning_normalized is not None:
            cleaning_normalized = np.concatenate((cleaning_normalized, [cleaning_normalized[0]]))
            ax.plot(angles, cleaning_normalized, 'o-', linewidth=2,
                    label=cleaning_label, color='green')
            ax.fill(angles, cleaning_normalized, alpha=0.2, color='green')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)

        max_val = max(P15_normalized.max(), 1.5)
        min_val = min(P15_normalized.min(), -0.5)
        ax.set_ylim([min_val, max_val])

        ax.set_title('雷达图对比：基线 -> 进样 -> 清洗\n归一化响应（相对基线的变化比例）',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"雷达图对比已保存至: {save_path}")

        plt.close()

        return self

    def visualize_all(self, save_dir=None):
        """生成所有可视化图表"""
        print(f"\n{'=' * 60}")
        print("正在生成所有可视化图表...")
        print(f"{'=' * 60}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        timeseries_path = None
        if save_dir:
            timeseries_path = os.path.join(save_dir,
                                           f"{self.filename.replace('.txt', '_timeseries.png')}")
        self.plot_full_timeseries(timeseries_path)

        baseline_box_path = None
        if save_dir:
            baseline_box_path = os.path.join(save_dir,
                                             f"{self.filename.replace('.txt', '_baseline_boxplot.png')}")
        self.plot_baseline_boxplot(baseline_box_path)

        response_path = None
        if save_dir:
            response_path = os.path.join(save_dir,
                                         f"{self.filename.replace('.txt', '_response_delta.png')}")
        self.plot_response_delta(response_path)

        corr_path = None
        if save_dir:
            corr_path = os.path.join(save_dir,
                                     f"{self.filename.replace('.txt', '_temporal_correlation.png')}")
        self.plot_temporal_correlation_heatmap(corr_path)

        radar_path = None
        if save_dir:
            radar_path = os.path.join(save_dir,
                                      f"{self.filename.replace('.txt', '_radar_comparison.png')}")
        self.plot_radar_comparison(radar_path)

        if save_dir:
            report_path = os.path.join(save_dir,
                                       f"{self.filename.replace('.txt', '_report.txt')}")
            self.save_text_report(report_path)

        print(f"\n{'=' * 60}")
        print("所有图表生成完成！")
        print(f"{'=' * 60}\n")

        return self

    def run_full_analysis(self, visualize=True, save_dir=None):
        """运行完整分析流程"""
        try:
            self.parse_file()
            self.analyze_environment()         # 硬件体检
            self.analyze_baseline()
            self.analyze_sampling()
            self.calculate_temporal_correlation()
            self.analyze_cleaning()
            self.calculate_quality_score()
            self.generate_report()

            if visualize:
                self.visualize_all(save_dir)

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
        print("Usage: python single.py <input_txt_path>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]

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

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"\n[ERROR] 错误：找不到文件")
        print(f"   文件路径: {file_path}")
        print(f"   当前工作目录: {os.getcwd()}")
        print(f"\n[TIP] 提示：")
        print(f"   1. 检查文件路径是否正确")
        print(f"   2. 使用绝对路径，如: E:/data/sample.txt")
        print(f"   3. 或使用相对路径，确保文件在当前目录下\n")
        sys.exit(1)

    print(f"\n[FILE] 正在分析文件: {os.path.basename(file_path)}")
    print(f"   完整路径: {os.path.abspath(file_path)}\n")

    # 运行分析
    eda = SingleFileEDA(file_path, skip_first_row=True)
    result = eda.run_full_analysis(
        visualize=True,
        save_dir=output_dir
    )

    if result:
        print("\n" + "=" * 60)
        print("[OK] 分析完成！")
        print("=" * 60)
        print(f"质量评分: {result.quality_score}/100")
        print(f"警告数量: {len(result.warnings)}")
        if result.responsive_sensors is not None:
            print(f"响应传感器: {len(result.responsive_sensors)}个")
        if 'cleaning_degree' in result.results:
            clean_stg = result.results.get('clean_stage', '清洗')
            print(f"{clean_stg}阶段清洗度: {result.results['cleaning_degree']:.1f}%")
        print(f"\n[FOLDER] 所有结果已保存到: {output_dir}/")
        print("=" * 60 + "\n")
