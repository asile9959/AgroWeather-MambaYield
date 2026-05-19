# AgroWeather-MambaYield

基于日尺度气象数据的农产品产量预测与气象因子影响分析项目。

当前模型围绕“气象数据 -> 作物产量预测 -> 影响解释”展开，核心包括气象-产量对齐、气象特征工程、多尺度时间卷积、LSTM-Attention、Mamba/Mamba2 分支、气象变量注意力和消融实验。

## 核心功能

- 气象特征工程：累计降水、平均温度、高温日数、连续少雨、GDD、复合胁迫等。
- 机器学习基线：Ridge、RandomForest、GradientBoosting 等。
- 深度模型：MSTC + LSTM-Attention + Mamba-compatible branch + ECA 气象变量注意力。
- 消融实验：对比 LSTM、MSTC、Mamba、ECA 及其组合是否有效。
- 可解释输出：气象变量注意力权重、预测值与真实值、分支融合权重。

## 主要文件

| 文件 | 说明 |
| --- | --- |
| `weather_yield_experiment.py` | 机器学习基线与气象特征工程实验 |
| `weather_yield_deep_experiment.py` | 深度学习主模型实验 |
| `run_weather_yield_ablations.py` | 批量消融实验脚本 |
| `实验运行说明与结果摘要.md` | 当前实验运行说明 |
| `气象产量预测完整模块化模型方案.md` | 模型设计方案 |

## 环境

基础实验环境：

```bash
pip install -r requirements.txt
```

如果要运行真正的 Mamba2，建议使用 WSL2/Linux + NVIDIA GPU + CUDA PyTorch，并额外安装：

```bash
pip install "causal-conv1d>=1.4.0" --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

如果 `mamba_ssm` 不可用，代码会自动使用 fallback 时序块，实验仍可运行。

## 快速运行

机器学习基线：

```bash
python weather_yield_experiment.py
```

深度模型：

```bash
python weather_yield_deep_experiment.py --epochs 20 --batch-size 128 --out-dir outputs_weather_yield_deep_mstc_eca
```

消融实验：

```bash
python run_weather_yield_ablations.py --epochs 20 --batch-size 128 --out-root outputs_weather_yield_ablations
```

日尺度数据正式实验示例：

```bash
python run_weather_yield_ablations.py \
  --epochs 50 \
  --batch-size 64 \
  --window-size 120 \
  --yield-path /path/to/yield.csv \
  --weather-path /path/to/weather_daily.csv \
  --out-root outputs_daily_ablations_w120
```

## 输出结果

消融实验汇总：

```text
outputs_weather_yield_ablations/ablation_summary.csv
```

深度模型指标：

```text
outputs_weather_yield_deep_mstc_eca/deep_test_metrics.csv
```

气象变量注意力：

```text
outputs_weather_yield_deep_mstc_eca/deep_variable_attention_summary.csv
```

## 数据说明

建议日尺度气象数据至少包含：

| 字段 | 说明 |
| --- | --- |
| `crop` | 作物类别 |
| `year` | 年份 |
| `date` | 气象日期 |
| `yield` | 产量 |
| `rain` | 日降水 |
| `tmax` | 日最高温 |
| `tmin` | 日最低温 |
| `tavg` | 日均温 |
| `humidity` | 湿度，可选 |

如有播种期、收获期、地区、土壤数据，可以继续扩展气象-产量对齐模块。
