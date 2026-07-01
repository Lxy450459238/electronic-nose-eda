# ========================================
# 电子鼻EDA通用配置 - 自动识别新旧版本
# ========================================
# 支持旧版（37传感器）和新版（49传感器）自动识别
# 修改时间：2026.06.08

import numpy as np


class UniversalConfig:
    """
    通用配置类 - 自动识别新旧版本电子鼻
    支持37传感器（旧版）和49传感器（新版）
    """

    # ========================================
    # 传感器版本配置
    # ========================================

    # 环境与系统通道范围校验参数（表9.1 硬件环境通道安全范围表）
    ENV_VALID_RANGES = {
        'VCC5V': (2375, 2750),       # 2500±5%，其中±5%为电源波动范围 (mV)
        'TEMP': (10, 80),            # 气室工作温度安全范围 (℃)
        'HUM': (10, 70),             # 湿度安全范围 (%)
        'PRESS': (850, 1020)         # 气压安全范围 (mbar)，85-102kPa (IEEE P2520.1)
    }

    # 旧版（37传感器）配置
    OLD_VERSION = {
        'name': '旧版37传感器',
        'sensor_count': 37,
        'sensor_config': {
            'original': (12, 49)      # sensor0-36: 第13-49列
        },

        'env_config': {
            'temp_hum_press': (5, 12),   # 7个：温度1,温度2,热电偶,湿度1,湿度2,压力1,压力2
            'vcc': (49, 51),             # 2个：VCC系统电压
        },

        'env_name_map': {
            0: '39_温度1', 1: '40_温度2', 2: '45_热电偶温度',
            3: '41_湿度1', 4: '42_湿度2', 5: '43_压力1', 6: '44_压力2',
            7: '37_VCC_+5V1', 8: '38_VCC5V-H1'
        },

        'min_columns': 51,

        'sensor_name_map': {
            0: "00_TGS813", 1: "01_TGS2610D", 2: "02_MS1100", 3: "03_TGS826", 4: "04_TGS2602",
            5: "05_TGS822", 6: "06_MG812", 7: "07_4S", 8: "08_4HS+", 9: "09_C6H6",
            10: "10_SMD1007", 11: "11_WSP2110", 12: "12_MP4", 13: "13_MP135A", 14: "14_TGS2620",
            15: "15_TGS2611E", 16: "16_AQ201", 17: "17_TGS8669", 18: "18_2M012", 19: "19_TGS2600",
            20: "20_MR516", 21: "21_H2S", 22: "22_NH3", 23: "23_4CH3SH", 24: "24_SMD1013",
            25: "25_SMD1001", 26: "26_MQ135", 27: "27_MP503", 28: "28_MQ3B", 29: "29_MQ137",
            30: "30_4ETO-10", 31: "31_4OXV", 32: "32_MQ138", 33: "33_MP901", 34: "34_ME3-C2H6S",
            35: "35_ME3-CH2O", 36: "36_PID-AH"
        }
    }

    # 新版（精简49纯气体传感器）配置
    NEW_VERSION = {
        'name': '精简版49纯气体传感器',
        'sensor_count': 49,
        'sensor_config': {
            'part1_0_36': (12, 49),       # 第13-49列 (对应原始 sensor0-36)
            'part2_44_47': (56, 60),      # 第57-60列 (对应 ME3-NH3, PID_AR 等)
            'part3_57_59': (69, 72),      # 第70-72列 (对应 TGS816, MP905 等)
            'part4_MICS_CO2': (76, 79),   # 第77-79列 (对应 MICS, CO2 等)
            'part5_ZE40': (81, 83),       # 第82-83列 (对应 ZE40 酒精传感器)
        },

        'env_config': {
            'temp_hum_press': (5, 12),    # 7个：温度1,温度2,热电偶,湿度1,湿度2,压力1,压力2
            'vcc': (49, 51),              # 2个：VCC系统电压
            'extra_temp_hum': (79, 81)    # 2个：扩展温湿度
        },

        'env_name_map': {
            0: '39_温度1', 1: '40_温度2', 2: '45_热电偶温度',
            3: '41_湿度1', 4: '42_湿度2', 5: '43_压力1', 6: '44_压力2',
            7: '37_VCC_+5V1', 8: '38_VCC5V-H1',
            9: 'Temperature_扩展', 10: 'Humidity_扩展'
        },

        'min_columns': 83,

        'sensor_name_map': {
            # Part 1: 前37个基础传感器 (索引 0-36)
            0: "00_TGS813", 1: "01_TGS2610D", 2: "02_MS1100", 3: "03_TGS826", 4: "04_TGS2602",
            5: "05_TGS822", 6: "06_MG812", 7: "07_4S", 8: "08_4HS+", 9: "09_C6H6",
            10: "10_SMD1007", 11: "11_WSP2110", 12: "12_MP4", 13: "13_MP135A", 14: "14_TGS2620",
            15: "15_TGS2611E", 16: "16_AQ201", 17: "17_TGS8669", 18: "18_2M012", 19: "19_TGS2600",
            20: "20_MR516", 21: "21_H2S", 22: "22_NH3", 23: "23_4CH3SH", 24: "24_SMD1013",
            25: "25_SMD1001", 26: "26_MQ135", 27: "27_MP503", 28: "28_MQ3B", 29: "29_MQ137",
            30: "30_4ETO-10", 31: "31_4OXV", 32: "32_MQ138", 33: "33_MP901", 34: "34_ME3-C2H6S",
            35: "35_ME3-CH2O", 36: "36_PID-AH",

            # Part 2: 扩展传感器1组 (索引 37-40)
            37: "44_AD_ME3-NH3", 38: "45_AD_ME2-CH2O", 39: "46_AD_ME3-H2S", 40: "47_AD_PID_AR",

            # Part 3: 扩展传感器2组 (索引 41-43)
            41: "57_AD_TGS816", 42: "58_AD_MP905", 43: "59_AD_TGS2612-D00",

            # Part 4: 光学/红外类气体传感器 (索引 44-46)
            44: "MICS5524_ADC", 45: "MH-Z14B_CO2", 46: "MH-Z19B_CO2",

            # Part 5: 酒精传感器 (索引 47-48)
            47: "ZE40_CH5OH", 48: "ZE40A_CH5OH"
        }
    }

    # ========================================
    # 分析参数配置（通用）
    # ========================================

    # --- 1. 基线分析参数 (四维融合判定规则) ---
    BASELINE_CV_STABLE = 0.15              # 变异系数(CV)上限
    BASELINE_MIN_MEAN_THRESHOLD = 50.0     # 均值绝对值门限
    BASELINE_MAX_STD_FALLBACK = 20.0       # 小均值通道的绝对标准差上限
    BASELINE_MAX_ALLOWED_STD = 30.0        # 全局STD红线
    BASELINE_MAX_PTP_DRIFT = 120.0         # 极差红线
    BASELINE_UNSTABLE_RATIO = 0.15         # 允许的不稳定传感器最大比例
    BASELINE_IQR_MULTIPLIER = 3.0          # 箱线图毛刺检测乘数
    BASELINE_OUTLIER_RATIO_THRESHOLD = 0.05  # 单通道容忍的异常点比例上限
    SCORE_BASELINE_OUTLIERS = 10            # 存在毛刺超标通道时的单次扣分（仅扣一次，不累计）
    CROSSED_BASELINE_MAX_RATIO = 0.15     # 反向越界（越过基线）传感器允许的最大比例（超过时发出警告）

    # --- 2. 稳态识别参数 ---
    STEADY_MIN_LENGTH = 30
    STEADY_MIN_LENGTH_CLEANING = 20
    STEADY_WINDOW_MAX_SIZE = 10
    STEADY_WINDOW_RATIO = 0.25
    STEADY_STABILITY_CV = 0.05
    STEADY_NEXT_SEGMENT = 20
    STEADY_FALLBACK_RATIO = 0.5
    STEADY_CLEANING_FALLBACK = 0.6
    STEADY_MAX_NO_STEADY = 10
    STEADY_CLEANING_WINDOW_RATIO = 0.2

    # --- 3. 响应分析参数 ---
    RESPONSE_RATIO_THRESHOLD = 0.1         # 进样变化幅度必须大于基线的10%
    RESPONSE_MIN_COUNT = 10                # 至少需要10个传感器产生有效响应
    RESPONSE_CORR_HIGH = 0.95              # 双气室进样高度一致的相关系数门限
    RESPONSE_CORR_LOW = 0.7                # 双气室进样严重分歧的警告门限

    # --- 4. 清洗度评估参数 ---
    CLEANING_EXCELLENT = 0.05              # 残留率<5%视为清洗优秀
    CLEANING_GOOD = 0.15                   # 残留率<15%视为清洗良好
    CLEANING_POOR = 0.25                   # 残留率≥25%视为恢复不足
    CLEANING_ACCEPTABLE_AVG = 0.15         # 整体平均残留率可接受上限
    CLEANING_GOOD_RATIO = 0.8              # 达到优/良标准的传感器比例
    CLEANING_EVAL_RESPONSIVE_ONLY = True   # 仅考核有响应的传感器

    # --- 5. 时间序列相关性分析参数 ---
    CORRELATION_HIGH_THRESHOLD = 0.8
    CORRELATION_DYNAMIC_RATIO = 0.5

    # --- 6. 质量评分参数 (基础分50，范围0-100) ---
    SCORE_BASE = 50
    SCORE_BASELINE_STABLE = 20
    SCORE_BASELINE_UNSTABLE = -15
    SCORE_BASELINE_OUTLIERS = -10
    SCORE_RESPONSE_GOOD = 25
    SCORE_RESPONSE_POOR = -15
    SCORE_P15_P17_CONSISTENT = 10
    SCORE_P15_P17_DIFFER = -10
    SCORE_CLEANING_EXCELLENT = 10
    SCORE_CLEANING_GOOD = 5
    SCORE_CLEANING_POOR = -10
    SCORE_EXCELLENT = 80                  # ≥80评为"优秀"
    SCORE_GOOD = 60                       # ≥60评为"良好"

    # --- 7. 多文件分析参数 ---
    STABILITY_CV_STABLE = 0.1
    STABILITY_CV_UNSTABLE = 0.3
    STABILITY_MIN_MEAN_THRESHOLD = 20.0   # 多文件低多样性：小均值场景阈值，|μ|≤20时切换绝对标准差判定
    STABILITY_MAX_STD_FALLBACK = 20.0      # 多文件低多样性：小均值场景绝对标准差上限，σ≤20判定为稳定

    DISCRIMINATION_SNR_WEIGHT = 0.6
    DISCRIMINATION_F_WEIGHT = 0.4
    DISCRIMINATION_SCORE_STRONG = 0.7
    DISCRIMINATION_SCORE_NORMAL = 0.5
    DISCRIMINATION_SCORE_ACCEPTABLE = 0.3
    DISCRIMINATION_P_VALUE_STRONG = 0.001
    DISCRIMINATION_P_VALUE_NORMAL = 0.01

    OUTLIER_IQR_MULTIPLIER = 1.5
    OUTLIER_MIN_ABNORMAL_SENSORS = 5
    CORRELATION_SAMPLE_COUNT = 10
    CORRELATION_MIN_POINTS = 10

    # --- 8. 可视化参数 ---
    VIS_FIGSIZE_TIMESERIES = (20, 10)
    VIS_FIGSIZE_BOXPLOT = (18, 6)
    VIS_FIGSIZE_HISTOGRAM = (18, 18)
    VIS_FIGSIZE_HISTOGRAM_MULTI = (21, 21)
    VIS_FIGSIZE_CORRELATION = (12, 10)
    VIS_FIGSIZE_RADAR = (12, 12)
    VIS_FIGSIZE_RADAR_MULTI = (16, 16)
    VIS_FIGSIZE_TABLE = (12, 8)
    VIS_DPI = 150
    VIS_DPI_HIGH = 200
    VIS_ALPHA_LINE = 0.7
    VIS_ALPHA_FILL = 0.2
    VIS_ALPHA_FILL_MULTI = 0.25
    VIS_ALPHA_STAGE = 0.2
    VIS_ALPHA_BOX = 0.7
    VIS_ALPHA_GRID = 0.3
    VIS_ALPHA_OUTLIER = 0.6
    VIS_HIST_BINS = 20
    VIS_MAX_OUTLIER_RADAR = 5

    # --- 9. 数据处理参数 ---
    DATA_SKIP_FIRST_ROW = True
    DATA_SKIP_ROWS_MAX = 3
    DATA_SKIP_CHECK_COLS = 37
    DATA_ENCODING = 'utf-8'
    DATA_SEPARATOR = ' '
    DATA_EPSILON = 1e-6
    PROGRESS_INTERVAL = 10

    # --- 10. 报告参数 ---
    REPORT_TOP_SENSORS = 15
    REPORT_TOP_PAIRS = 15

    # --- 11. 文件路径配置 ---
    FONT_PATH = 'C:/Windows/Fonts/simhei.ttf'

    @classmethod
    def detect_version(cls, data_columns):
        """自动检测电子鼻版本"""
        if data_columns >= cls.NEW_VERSION['min_columns']:
            print(f"[OK] 检测到新版电子鼻 ({cls.NEW_VERSION['sensor_count']}个传感器)")
            return cls.NEW_VERSION
        elif data_columns >= cls.OLD_VERSION['min_columns']:
            print(f"[OK] 检测到旧版电子鼻 ({cls.OLD_VERSION['sensor_count']}个传感器)")
            return cls.OLD_VERSION
        else:
            raise ValueError(
                f"数据文件列数不足！\n"
                f"  当前列数: {data_columns}\n"
                f"  旧版至少需要: {cls.OLD_VERSION['min_columns']}列\n"
                f"  新版至少需要: {cls.NEW_VERSION['min_columns']}列"
            )

    @classmethod
    def extract_sensor_data(cls, data_df, version_config):
        """根据版本配置提取传感器数据"""
        sensor_parts = []

        print("\n[读取] 正在读取传感器数据...")
        for config_name, (start, end) in version_config['sensor_config'].items():
            part = data_df.iloc[:, start:end].values
            sensor_parts.append(part)
            sensor_count = end - start
            print(f"   [OK] {config_name}: 列{start + 1}-{end} -> {sensor_count}个传感器")

        sensor_data = np.concatenate(sensor_parts, axis=1)
        print(f"\n[OK] 传感器数据合并完成: {sensor_data.shape} (样本数 x 传感器数)")

        expected_sensors = version_config['sensor_count']
        if sensor_data.shape[1] != expected_sensors:
            raise ValueError(
                f"传感器数量不匹配！期望 {expected_sensors}，实际 {sensor_data.shape[1]}"
            )

        return sensor_data

    @classmethod
    def get_sensor_name(cls, sensor_idx, version_config):
        """获取传感器名称"""
        return version_config['sensor_name_map'].get(sensor_idx, f'sensor{sensor_idx}')

    @classmethod
    def print_config_summary(cls, version_config):
        """打印配置摘要"""
        print(f"=" * 80)
        print(f"电子鼻EDA分析系统 - 配置信息")
        print(f"=" * 80)
        print(f"检测版本: {version_config['name']}")
        print(f"传感器数量: {version_config['sensor_count']}个")
        print(f"基线CV阈值: {cls.BASELINE_CV_STABLE} (稳定)")
        print(f"响应阈值: {cls.RESPONSE_RATIO_THRESHOLD} ({cls.RESPONSE_RATIO_THRESHOLD * 100:.0f}%)")
        print(f"清洗优秀标准: 残留<{cls.CLEANING_EXCELLENT * 100:.0f}%")
        print(f"评分优秀: >={cls.SCORE_EXCELLENT}分")
        print(f"=" * 80)

    @classmethod
    def extract_env_data(cls, data_df, version_config):
        """单独提取环境与物理通道数据"""
        if 'env_config' not in version_config:
            return None
        print(f"\n[读取] 正在读取环境与系统通道数据...")
        env_parts = []
        for config_name, (start, end) in version_config['env_config'].items():
            part = data_df.iloc[:, start:end].values
            env_parts.append(part)
            print(f"   [OK] {config_name}: 列{start + 1}-{end} -> {end - start}个监控通道")

        return np.concatenate(env_parts, axis=1)
