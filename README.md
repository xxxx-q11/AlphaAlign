# AlphaAlign
This document helps you complete two things before running AlphaAlign locally:

1. Prepare a data directory that has already been converted to Qlib format.
2. Fill that directory path into `provider_uri`, and configure the Python and LLM environments.

## 1. Prepare the Qlib Data Directory

The current project's Qlib backtesting service uses the China market configuration `REG_CN`, and the default backtest stock pool is `csi300`. Prepare an A-share Qlib-format data directory, for example:

```text
C:\qlib_data\cn_data
```

Or:

```text
~/.qlib/qlib_data/cn_data
```

This directory should directly contain the Qlib data folders, not their parent directory. A common structure looks like this:

```text
cn_data/
  calendars/
    day.txt
  instruments/
    all.txt
    csi300.txt
  features/
    sh600000/
      close.day.bin
      open.day.bin
      volume.day.bin
    sz000001/
      close.day.bin
      open.day.bin
      volume.day.bin
```

## 2. Fill in `provider_uri`

The project's default backtest configuration file is:

```text
config/backtest_default.yaml
```

Open this file and find:

```yaml
provider_uri: null
```

Change `null` to your Qlib data directory. For Windows paths, it is recommended to use forward slashes or wrap the path in quotes:

```yaml
provider_uri: "~/.qlib/qlib_data/cn_data"
```

## 3. Configure the Python Environment

It is recommended to use an isolated virtual environment to avoid mixing project dependencies with the system Python installation.

PowerShell example:

```powershell
pip install -r requirement.txt
```

After installation, verify that Qlib can be imported:

```powershell
python -c "import qlib; print(qlib.__version__)"
```

## 4. Configure the LLM Environment

The main workflow requires an LLM configuration. The project provides an example file:

```text
config/env.example.yaml
```

Copy it as your local configuration:

```linux
cp config/env.example.yaml config/env.yaml
```

Then edit `config/env.yaml`:

```yaml
provider: "third_party"
api_key: "YOUR_API_KEY"
model: "deepseek-v3.2"
base_url: "https://openrouter.ai/api/v1"
temperature: 0.7
max_tokens: null
```

Notes:

- `config/env.yaml` contains secrets and should only be used for local runs.
- This file is already ignored in `.gitignore`; do not commit a real API key.
- If you only run the standalone backtest without enabling news review, an LLM is usually not required. Running the complete `main.py` workflow requires it.

## 5. Pre-Run Check

Use a short Python command to initialize Qlib directly and read the stock pool:

```powershell
python -c "import qlib; from qlib.config import REG_CN; from qlib.data import D; qlib.init(provider_uri='C:/qlib_data/cn_data', region=REG_CN); print(D.instruments('csi300'))"
```

If the stock pool configuration is printed normally, `provider_uri` is basically usable.

## 6. Run the Project

Run the complete AlphaAlign workflow:

```powershell
python main.py
```

Use a custom backtest configuration file:

```powershell
python main.py --backtest-config-path config/backtest_default.yaml
```

Run the standalone Qlib backtest:

You can copy the two files under `data\case` into `data`, then start the standalone backtest.
`run_linear_weighting_qlib_backtest.py` reads `data\factor_library.json` under `data` by default for backtesting. `python main.py` does the same.

```powershell
python run_linear_weighting_qlib_backtest.py   --signal-mode rolling   --selector-mode mwu   --weighting-method mwu   --start-date 2024-01-01   --end-date 2026-01-01   --window-days 90   --rebalance-window-days 10   --n-drop 5   --daily-buy-topk 5   --factor-eval-top-k 5   --mwu-learning-rate 0.15   --mwu-reward-cap 0.05   --mwu-explore-rate 0.03   --mwu-max-weight 0.15
```

### Parameter Description

- `--window-days 90`  
  The time window used by MWU to evaluate factors. It evaluates each factor using the excess returns of the top `top_K` stocks from 100 trading days ago to 10 trading days ago.

- `--rebalance-window-days 10`  
  The MWU weight holding period. MWU weights are recalculated every 10 trading days.

- `--mwu-learning-rate 0.15`  
  The MWU learning rate, which controls how sensitive the weights are to recent return feedback. A larger value makes weights adjust faster, but also makes them more likely to chase rallies and sell into declines.

- `--mwu-reward-cap 0.05`  
  The clipping upper bound for a single day's expert excess return. Here it means returns are clipped to `[-5%, +5%]` before being used for weight updates, preventing extreme returns from dominating the results.

- `--mwu-explore-rate 0.03`  
  The exploration rate. After each update, `3%` uniform weight is mixed in to prevent weights from becoming overly concentrated in a small number of factors.

- `--mwu-max-weight 0.15`  
  The maximum weight limit for a single expert. Here it means each factor-direction expert can have a weight of at most `15%`, which controls portfolio concentration.

Backtest outputs are written to:

```text
data/backtest_outputs/
```
