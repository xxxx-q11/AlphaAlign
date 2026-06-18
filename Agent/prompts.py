"""
Prompt template module
Stores prompt templates used by various Agents
"""

# ==================== AlphaSAGE Valid Operator Definitions ====================
# Must be consistent with Qlib_MCP/workspace/AlphaSAGE/src/alphagen/config.py

# Valid feature names (lowercase)
VALID_FEATURES = {"open_", "close", "high", "low", "volume", "vwap"}

# Valid operator names
VALID_OPERATORS = {
    # Unary operators
    "Abs", "Log", "SLog1p", "Sign", "Inv",
    # Binary operators
    "Add", "Sub", "Mul", "Div", "Pow", "Greater", "Less",
    # Time series operators
    "Ref", "TsMean", "TsSum", "TsStd", "TsVar", "TsMax", "TsMin", "TsMed",
    "TsRank", "TsDelta", "TsDiv", "TsSkew", "TsKurt", "TsWMA", "TsEMA",
    "TsPctChange", "TsIr", "TsMinMaxDiff", "TsMaxDiff", "TsMinDiff", "TsMad",
    # Pair time series operators
    "TsCov", "TsCorr"
}

# Valid constants
VALID_CONSTANTS = {-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.}

# Valid time windows
VALID_WINDOWS = {10, 20, 30, 40, 50}

# Operator description text (for prompt use)
OPERATORS_DESCRIPTION = """### Features
- open_: Opening price
- close: Closing price
- high: Highest price
- low: Lowest price
- volume: Trading volume
- vwap: Volume weighted average price

### Unary Operators
- Abs(x): Absolute value
- Log(x): Logarithm
- SLog1p(x): Safe logarithm log(1+x)
- Sign(x): Sign function
- Inv(x): Reciprocal 1/x

### Binary Operators
- Add(x, y): Addition x + y
- Sub(x, y): Subtraction x - y
- Mul(x, y): Multiplication x * y
- Div(x, y): Division x / y
- Pow(x, y): Power x^y
- Greater(x, y): x > y ? 1 : 0
- Less(x, y): x < y ? 1 : 0

### Time Series Operators (Rolling Operators) - require window parameter
- Ref(x, d): Value d days ago
- TsMean(x, d): d-day mean
- TsSum(x, d): d-day sum
- TsStd(x, d): d-day standard deviation
- TsVar(x, d): d-day variance
- TsMax(x, d): d-day maximum
- TsMin(x, d): d-day minimum
- TsMed(x, d): d-day median
- TsRank(x, d): d-day rank
- TsDelta(x, d): d-day change
- TsDiv(x, d): Current value divided by value d days ago
- TsSkew(x, d): d-day skewness
- TsKurt(x, d): d-day kurtosis
- TsWMA(x, d): d-day weighted moving average
- TsEMA(x, d): d-day exponential moving average
- TsPctChange(x, d): d-day percentage change
- TsIr(x, d): d-day information ratio (mean/std)
- TsMinMaxDiff(x, d): d-day max-min difference
- TsMaxDiff(x, d): d-day maximum difference from current value
- TsMinDiff(x, d): d-day minimum difference from current value
- TsMad(x, d): d-day mean absolute deviation

### Pair Time Series Operators (Pair Rolling Operators)
- TsCov(x, y, d): d-day covariance
- TsCorr(x, y, d): d-day correlation coefficient

### Available Constants
-30, -10, -5, -2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2, 5, 10, 30

### Recommended Time Windows
10, 20, 30, 40, 50"""


# Factor mining related prompts
FACTOR_MINING_SYSTEM_PROMPT = """You are a quantitative code analysis expert, skilled at extracting and analyzing quantitative trading strategies from code.
Your tasks are:
1. Carefully analyze the provided code and documentation
2. Identify and extract quantitative trading strategy definitions
3. Understand the strategy's calculation logic and purpose
4. Output strategy information in a structured format"""

FACTOR_MINING_ANALYSIS_PROMPT = """Please analyze the quantitative trading strategies in the following repository:

Repository path: {repo_path}

README document content:
{readme_content}

Training script content:
{training_script_content}

The current script may not be the algorithm strategy proposed by the repository, but rather a baseline script for comparison. You need to obtain the following information from the current script:
1. Strategy name and execution logic
2. Parameters accepted by the training script when started via terminal command and their meanings
3. Command to start the training script
4. Data path where the factor pool file is saved after the training script runs
5. Data format of the factor pool file saved after the training script runs

Please output in JSON format:
```json

{{
    "strategy_name": "Strategy name",
    "strategy_description": "Strategy description, which years the strategy trains between when started with default terminal command",
    "strategy_parameters": {{"key_parameters":"Parameter name","value_parameters_scope":"Parameter range","value_parameters_default":"Parameter default value","description":"Parameter description"}},
    "training_command": "Command to start the training script",
    "strategy_data_path": "Data path where factor pool file is saved"
    "strategy_data_format": "Data format of factor pool file saved"
}}
```
"""


# Factor template generation prompt (for LLM to generate GP seed factors based on feedback)
FACTOR_TEMPLATE_GENERATION_PROMPT = """You are a quantitative factor engineering expert, skilled at designing high-quality Alpha factor expressions.

## Task
Based on the following feedback information and available operators, generate 20-30 factor template expressions to guide factor mining using Genetic Programming (GP) algorithm.

## Available Operators and Features (strictly follow the definitions below, do not use other operators)

""" + OPERATORS_DESCRIPTION + """

## Current Feedback Information

### Current Iteration Round
{iteration}

### Current Factor Pool Weaknesses
{pool_weaknesses}

### Suggested Exploration Directions
{suggested_directions}

### GP Strategy Hints
{gp_strategy_hints}

### Convergence Information
{convergence_info}

### Existing High-Quality Factors (for reference, avoid duplicates)
{existing_seeds}

## Output Requirements

Please generate factor template expressions following these rules:
1. Use the operators and features defined above
2. Expression format examples:
   - `Div(TsMean(close, 10), TsMean(close, 30))`
   - `Sub(TsMax(high, 20), TsMin(low, 20))`
   - `TsCorr(close, volume, 30)`
3. Design complementary factors targeting weaknesses in feedback
4. Explore directions suggested in feedback
5. Each factor should have diversity, covering different types (momentum, volatility, volume, price pattern, etc.)

Please output in JSON format:
```json
{{
    "factor_templates": [
        {{
            "expression": "Factor expression",
            "category": "Factor type (momentum/volatility/volume/price_pattern/correlation/mean_reversion)",
            "description": "Brief description of the factor design rationale",
            "addresses_weakness": "Which weakness this factor addresses (optional)"
        }}
    ],
    "generation_strategy": "Generation strategy explanation, explaining why these factors were chosen"
}}
```
"""


# Factor pool analysis and seed suggestion prompt (for factor_eval_agent use)
FACTOR_POOL_ANALYSIS_PROMPT = """You are a senior quantitative factor research expert. Please complete two tasks based on the following factor pool analysis report:

{pool_report}

---

## Available Operators and Features (strictly follow the definitions below, generated factor expressions can only use these operators)

""" + OPERATORS_DESCRIPTION + """

---

Please complete the following analysis and output JSON:

1. **Analyze current factor pool weaknesses**:
   - Use both factor expressions and their economics explanations from the report.
   - Are factor types too homogeneous? (e.g., all momentum type, lacking volatility type)
   - Are window periods too concentrated? (e.g., all short-term, lacking medium/long-term)
   - Is feature usage too homogeneous? (e.g., only using close, lacking volume, high/low, etc.)

2. **Analyze which factors improved returns**:
   - Identify factor characteristics with greatest contribution from attribution analysis

3. **Generate seed factor suggestions**:
   - Generate 10-20 differentiated seed factors targeting weaknesses
   - **Strictly use the operators and features defined above, do not use other operators**
   - Feature names: open_, close, high, low, volume, vwap
   - Expression example: Div(TsMean(close, 10), TsMean(close, 30))

Output JSON format:
```json
{{
    "pool_weaknesses": [
        "Weakness 1: specific description",
        "Weakness 2: specific description"
    ],
    "effective_patterns": [
        "Effective pattern 1: identified from attribution analysis"
    ],
    "suggested_directions": [
        {{
            "direction": "Direction description",
            "reason": "Why this direction is needed",
            "priority": "high/medium/low"
        }}
    ],
    "suggested_seeds": [
        {{
            "expression": "Factor expression (using operators defined above)",
            "category": "Factor category (e.g.: volatility/momentum/liquidity)",
            "expected_effect": "Expected effect"
        }}
    ],
    "gp_strategy_hints": {{
        "preferred_operators": ["Recommended operators to use more"],
        "preferred_features": ["Recommended features to use more"],
        "preferred_windows": [Recommended window sizes],
        "avoid_patterns": ["Patterns to avoid"]
    }}
}}
```

Only output JSON, no other content.
"""


# More types of prompts can be added
# For example:
# CODE_ANALYSIS_PROMPT = "..."
# DOCUMENTATION_PROMPT = "..."
# etc.
