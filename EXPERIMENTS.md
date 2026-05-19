# Experiments

## Main Scripts

- `weather_yield_experiment.py`: weather-yield alignment, weather feature engineering, machine-learning baselines, and basic factor-importance analysis.
- `weather_yield_deep_experiment.py`: MSTC + LSTM-Attention + Mamba/Mamba2-compatible branch + ECA weather-variable attention.
- `run_weather_yield_ablations.py`: batch ablation runner for LSTM, MSTC, Mamba, ECA, and combined models.

## Engineered Weather Features

- `RainSum`: cumulative rainfall.
- `TavgMean`: average temperature.
- `HotDays`: high-temperature days or proxy.
- `DrySpell`: consecutive low-rainfall days or proxy.
- `GDD`: growing degree days.
- `Stress`: combined heat and drought stress.
- `HotStress`: heat stress.
- `DroughtStress`: drought stress.
- `HumidityMean`: mean humidity when available.

## Quick Commands

Machine-learning baselines:

```bash
python weather_yield_experiment.py
```

Deep model:

```bash
python weather_yield_deep_experiment.py --epochs 20 --batch-size 128 --out-dir outputs_weather_yield_deep_mstc_eca
```

Ablations:

```bash
python run_weather_yield_ablations.py --epochs 20 --batch-size 128 --out-root outputs_weather_yield_ablations
```

Daily-data formal experiment:

```bash
python run_weather_yield_ablations.py \
  --epochs 50 \
  --batch-size 64 \
  --window-size 120 \
  --yield-path /path/to/yield.csv \
  --weather-path /path/to/weather_daily.csv \
  --out-root outputs_daily_ablations_w120
```

## Current Local Findings

On the current annual-aggregate sample data, ablation results showed MSTC and ECA were the most stable additions. The Mamba branch should be judged again after daily weather data and a real `mamba_ssm.Mamba2` environment are available.
