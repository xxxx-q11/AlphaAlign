"""
AlphaSAGE Expression -> Qlib Expression Converter

Converts AlphaSAGE PyTorch-based factor expressions (expression.py) into
Qlib-compatible expression string format.

Usage:
    from alphagen.utils.qlib_converter import AlphaSAGEToQlibConverter

    converter = AlphaSAGEToQlibConverter()

    # Single expression conversion
    qlib_expr = converter.convert("TsIr(TsMean(volume,30),50)")
    # Output: "Div(Mean($volume, 30), Std($volume, 30))"

    # Batch conversion
    qlib_exprs = converter.convert_batch([
        {"expression": "TsIr(TsMean(volume,30),50)", "ic": 0.044},
        {"expression": "TsMin(close,20)", "ic": 0.035}
    ])
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path


class AlphaSAGEToQlibConverter:
    """AlphaSAGE expression to Qlib expression converter"""

    # Feature name mapping: AlphaSAGE -> Qlib
    FEATURE_MAP = {
        'close': '$close',
        'Close': '$close',
        'CLOSE': '$close',
        'open_': '$open',
        'Open': '$open',
        'OPEN': '$open',
        'high': '$high',
        'High': '$high',
        'HIGH': '$high',
        'low': '$low',
        'Low': '$low',
        'LOW': '$low',
        'volume': '$volume',
        'Volume': '$volume',
        'VOLUME': '$volume',
        'vwap': '$vwap',
        'Vwap': '$vwap',
        'VWAP': '$vwap',
    }

    # Unary operator mapping (direct correspondence)
    UNARY_OP_MAP = {
        'Abs': 'Abs',
        'Sign': 'Sign',
        'Log': 'Log',
    }

    # Unary operators that need expansion
    UNARY_OP_EXPAND = {
        'SLog1p': lambda x: f'Mul(Sign({x}), Log(Add(Abs({x}), 1)))',
        'Inv': lambda x: f'Div(1, {x})',
    }

    # Cross-sectional operators - these cannot be directly implemented at the qlib expression layer.
    # They need to be implemented in the data processing layer via processors (e.g., CSRankNorm).
    # When converting, return the inner expression directly and mark it as needing cross-sectional processing.
    CROSS_SECTIONAL_OPS = {'Rank'}

    # Binary operator mapping
    BINARY_OP_MAP = {
        'Add': 'Add',
        'Sub': 'Sub',
        'Mul': 'Mul',
        'Div': 'Div',
        'Pow': 'Power',
        'Greater': 'Greater',
        'Less': 'Less',
    }

    # Rolling operator mapping (direct correspondence)
    ROLLING_OP_MAP = {
        'Ref': 'Ref',
        'TsMean': 'Mean',
        'TsSum': 'Sum',
        'TsStd': 'Std',
        'TsVar': 'Var',
        'TsSkew': 'Skew',
        'TsKurt': 'Kurt',
        'TsMax': 'Max',
        'TsMin': 'Min',
        'TsMed': 'Med',
        'TsMad': 'Mad',
        'TsRank': 'Rank',
        'TsDelta': 'Delta',
        'TsWMA': 'WMA',
        'TsEMA': 'EMA',
    }

    # Rolling operators that need expansion (return a function taking expression and window size)
    ROLLING_OP_EXPAND = {
        # TsIr = mean / std
        'TsIr': lambda x, n: f'Div(Mean({x}, {n}), Std({x}, {n}))',
        # TsMinMaxDiff = max - min
        'TsMinMaxDiff': lambda x, n: f'Sub(Max({x}, {n}), Min({x}, {n}))',
        # TsMaxDiff = current - max (current value minus window maximum)
        'TsMaxDiff': lambda x, n: f'Sub({x}, Max({x}, {n}))',
        # TsMinDiff = current - min (current value minus window minimum)
        'TsMinDiff': lambda x, n: f'Sub({x}, Min({x}, {n}))',
        # TsDiv = current / mean (current value divided by window mean)
        'TsDiv': lambda x, n: f'Div({x}, Mean({x}, {n}))',
        # TsPctChange = (current - oldest) / oldest
        # Note: Ref(x, N-1) is the value from N-1 days ago
        'TsPctChange': lambda x, n: f'Div(Sub({x}, Ref({x}, {int(n)-1})), Ref({x}, {int(n)-1}))',
    }

    # Pair Rolling operator mapping
    PAIR_ROLLING_OP_MAP = {
        'TsCov': 'Cov',
        'TsCorr': 'Corr',
    }

    # Qlib Rolling operator list (first argument must be an expression, not a constant)
    QLIB_ROLLING_OPS = {
        'Mean', 'Sum', 'Std', 'Var', 'Skew', 'Kurt', 'Max', 'Min',
        'Med', 'Mad', 'Rank', 'Delta', 'WMA', 'EMA', 'Ref'
    }

    # Qlib Pair Rolling operator list
    QLIB_PAIR_ROLLING_OPS = {'Cov', 'Corr'}

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the converter

        Args:
            strict_mode: In strict mode, exceptions are raised for unrecognized operators
        """
        self.strict_mode = strict_mode
        # Track whether the current expression contains cross-sectional operators
        self._has_cross_sectional_op = False
        # Track whether the expression is valid
        self._is_valid_expression = True
        self._invalid_reason = None

    def convert(self, alphagen_expr: str) -> str:
        """
        Convert an AlphaSAGE expression to a Qlib expression

        Args:
            alphagen_expr: Expression string in AlphaSAGE format

        Returns:
            Expression string in Qlib format

        Example:
            >>> converter = AlphaSAGEToQlibConverter()
            >>> converter.convert("TsIr(TsMean(volume,30),50)")
            'Div(Mean(Mean($volume, 30), 50), Std(Mean($volume, 30), 50))'
        """
        # Reset state flags
        self._has_cross_sectional_op = False
        self._is_valid_expression = True
        self._invalid_reason = None

        # Strip whitespace
        expr = alphagen_expr.strip()

        # Recursively convert (returns the expression and a const-ness flag)
        result, _ = self._convert_recursive(expr)

        # Validation is already done during recursive conversion; no separate _validate_qlib_expression call needed

        return result

    def _validate_qlib_expression_regex(self, qlib_expr: str):
        """
        [Fallback method] Validate converted Qlib expression using regex

        Note: This method can only detect direct constant arguments, not nested constant expressions.
        The main validation logic has been moved to constant-ness tracking in _convert_recursive.

        Detects invalid patterns:
        - First argument of a Rolling operator is a direct constant (e.g., Med(-0.01, 10))
        - Arguments of a Pair Rolling operator are direct constants

        Args:
            qlib_expr: Expression string in Qlib format
        """
        # Build regex pattern for Rolling operators
        # Matches: OpName(literal number, number) - e.g., Med(-0.01, 10)
        rolling_ops_pattern = '|'.join(self.QLIB_ROLLING_OPS)
        # Matches negative numbers, decimals, integers: -0.01, 5.0, 10, -5, etc.
        number_pattern = r'-?\d+\.?\d*'

        # Detect cases where the first arg of a Rolling operator is a constant
        invalid_rolling_pattern = rf'({rolling_ops_pattern})\(\s*({number_pattern})\s*,\s*\d+\s*\)'
        match = re.search(invalid_rolling_pattern, qlib_expr)
        if match:
            self._is_valid_expression = False
            self._invalid_reason = f"Rolling operator {match.group(1)} has constant {match.group(2)} as its first argument; qlib does not support rolling operations on constants"
            return

        # Detect cases where Pair Rolling operator arguments are constants
        pair_rolling_ops_pattern = '|'.join(self.QLIB_PAIR_ROLLING_OPS)

        # First argument is constant: Cov(-0.01, expr, n)
        invalid_pair_first = rf'({pair_rolling_ops_pattern})\(\s*({number_pattern})\s*,'
        match = re.search(invalid_pair_first, qlib_expr)
        if match:
            self._is_valid_expression = False
            self._invalid_reason = f"Pair Rolling operator {match.group(1)} has constant {match.group(2)} as its first argument; qlib does not support this"
            return

        # Second argument is constant: Cov(expr, -0.01, n)
        # Matching pattern is more complex, needs to skip the first argument
        invalid_pair_second = rf'({pair_rolling_ops_pattern})\([^,]+,\s*({number_pattern})\s*,\s*\d+\s*\)'
        match = re.search(invalid_pair_second, qlib_expr)
        if match:
            self._is_valid_expression = False
            self._invalid_reason = f"Pair Rolling operator {match.group(1)} has constant {match.group(2)} as its second argument; qlib does not support this"
            return

    def is_valid(self) -> bool:
        """
        Check whether the most recently converted expression is valid

        Returns:
            Whether the expression is valid
        """
        return self._is_valid_expression

    def get_invalid_reason(self) -> Optional[str]:
        """
        Get the reason why the expression is invalid

        Returns:
            The invalidity reason, or None if the expression is valid
        """
        return self._invalid_reason

    def convert_with_metadata(self, alphagen_expr: str) -> Dict[str, Any]:
        """
        Convert an AlphaSAGE expression to a Qlib expression and return metadata

        Args:
            alphagen_expr: Expression string in AlphaSAGE format

        Returns:
            Dictionary containing fields such as qlib_expression, needs_cs_rank, is_valid
        """
        qlib_expr = self.convert(alphagen_expr)
        result = {
            'qlib_expression': qlib_expr if self._is_valid_expression else None,
            'needs_cs_rank': self._has_cross_sectional_op,
            'is_valid': self._is_valid_expression
        }
        if not self._is_valid_expression:
            result['invalid_reason'] = self._invalid_reason
        return result

    def check_needs_cs_rank(self, alphagen_expr: str) -> bool:
        """
        Check whether the expression contains a cross-sectional Rank operator

        Args:
            alphagen_expr: Expression string in AlphaSAGE format

        Returns:
            Whether cross-sectional Rank processing is needed
        """
        self.convert(alphagen_expr)
        return self._has_cross_sectional_op

    def _convert_recursive(self, expr: str) -> Tuple[str, bool]:
        """
        Recursively convert an expression

        Returns:
            (converted expression, whether it is a pure constant expression)

        A pure constant expression is one that contains no time-series data
        (such as $close, $volume, etc.). Performing rolling operations on
        constant expressions is invalid in qlib.
        """
        expr = expr.strip()

        # Check if it is a Constant
        if expr.startswith('Constant('):
            # Constant(1.0) -> 1.0
            match = re.match(r'Constant\(([^)]+)\)', expr)
            if match:
                return match.group(1), True  # constant

        # Check if it is a pure feature name -> not a constant (it is time-series data)
        if expr in self.FEATURE_MAP:
            return self.FEATURE_MAP[expr], False

        # Check if it is a pure number -> constant
        try:
            float(expr)
            return expr, True
        except ValueError:
            pass

        # Check if fully wrapped in parentheses; if so, unwrap and re-convert.
        # This correctly handles cases like (Sub(TsRank(...))) with extra parentheses.
        if expr.startswith('(') and expr.endswith(')'):
            depth = 0
            fully_wrapped = True
            for i, char in enumerate(expr):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    # If depth becomes 0 before the last character, it is not fully wrapped
                    if depth == 0 and i < len(expr) - 1:
                        fully_wrapped = False
                        break
            if fully_wrapped and depth == 0:
                # Fully wrapped in parentheses; unwrap and re-convert
                return self._convert_recursive(expr[1:-1])

        # Parse operator and arguments
        parsed = self._parse_operator(expr)
        if parsed is None:
            # Cannot parse; may be a pure feature name or constant
            if expr in self.FEATURE_MAP:
                return self.FEATURE_MAP[expr], False
            # Unrecognized expression; conservatively treat as constant
            return expr, True

        op_name, args = parsed

        # Recursively convert arguments and track const-ness of each
        converted_args = []
        args_is_constant = []
        for arg in args:
            conv_arg, is_const = self._convert_recursive(arg)
            converted_args.append(conv_arg)
            args_is_constant.append(is_const)

        # Convert operator while validating const-ness
        return self._convert_operator_with_validation(op_name, converted_args, args_is_constant)

    def _parse_operator(self, expr: str) -> Optional[Tuple[str, List[str]]]:
        """
        Parse an expression, extracting the operator name and arguments

        Args:
            expr: Expression string, e.g., "TsMean(volume,30)"

        Returns:
            (operator_name, [argument_list]) or None
        """
        # Match operator name
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\((.+)\)$', expr)
        if not match:
            return None

        op_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments (handling nested parentheses)
        args = self._split_args(args_str)

        return op_name, args

    def _split_args(self, args_str: str) -> List[str]:
        """
        Split an argument string, handling nested parentheses

        Args:
            args_str: Argument string, e.g., "TsMean(volume,30),50"

        Returns:
            Argument list, e.g., ["TsMean(volume,30)", "50"]
        """
        args = []
        current_arg = []
        depth = 0

        for char in args_str:
            if char == '(':
                depth += 1
                current_arg.append(char)
            elif char == ')':
                depth -= 1
                current_arg.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current_arg).strip())
                current_arg = []
            else:
                current_arg.append(char)

        if current_arg:
            args.append(''.join(current_arg).strip())

        return args

    def _convert_operator_with_validation(
        self,
        op_name: str,
        args: List[str],
        args_is_constant: List[bool]
    ) -> Tuple[str, bool]:
        """
        Convert a single operator while validating that Rolling operator arguments are valid.

        Core logic: Track the "const-ness" of each sub-expression. When the first argument
        of a Rolling operator is a constant expression, mark the entire expression as invalid.

        Args:
            op_name: Operator name
            args: Converted argument list
            args_is_constant: Whether each argument is a constant expression

        Returns:
            (converted expression, whether this expression is a constant)
        """
        # Cross-sectional operators - preserve the const-ness of sub-expressions
        if op_name in self.CROSS_SECTIONAL_OPS:
            self._has_cross_sectional_op = True
            return args[0], args_is_constant[0]

        # Unary operators - preserve const-ness
        if op_name in self.UNARY_OP_MAP:
            qlib_op = self.UNARY_OP_MAP[op_name]
            return f'{qlib_op}({args[0]})', args_is_constant[0]

        # Unary operator expansion - preserve const-ness
        if op_name in self.UNARY_OP_EXPAND:
            expand_func = self.UNARY_OP_EXPAND[op_name]
            return expand_func(args[0]), args_is_constant[0]

        # Binary operators - result is constant only if both are constant
        if op_name in self.BINARY_OP_MAP:
            qlib_op = self.BINARY_OP_MAP[op_name]
            result_is_constant = args_is_constant[0] and args_is_constant[1]
            return f'{qlib_op}({args[0]}, {args[1]})', result_is_constant

        # Rolling operators - validate that the first argument is not a constant
        if op_name in self.ROLLING_OP_MAP:
            qlib_op = self.ROLLING_OP_MAP[op_name]
            if args_is_constant[0]:  # First argument is constant -> invalid!
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Rolling operator {qlib_op} has constant '{args[0]}' "
                    f"as its first argument; qlib does not support rolling operations on constants"
                )
            # Rolling operator results are never constant (they depend on time series;
            # even operating on a constant produces a series)
            return f'{qlib_op}({args[0]}, {args[1]})', False

        # Rolling operator expansion - same validation needed
        if op_name in self.ROLLING_OP_EXPAND:
            if args_is_constant[0]:
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Rolling operator {op_name} has constant '{args[0]}' "
                    f"as its first argument; qlib does not support rolling operations on constants"
                )
            expand_func = self.ROLLING_OP_EXPAND[op_name]
            return expand_func(args[0], args[1]), False

        # Pair Rolling operators - validate first two arguments
        if op_name in self.PAIR_ROLLING_OP_MAP:
            qlib_op = self.PAIR_ROLLING_OP_MAP[op_name]
            # Check if both time-series arguments are constants
            if args_is_constant[0] and args_is_constant[1]:
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Pair Rolling operator {qlib_op} has both arguments as constant expressions; qlib does not support this"
                )
            elif args_is_constant[0]:
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Pair Rolling operator {qlib_op} has constant '{args[0]}' "
                    f"as its first argument; qlib does not support this"
                )
            elif args_is_constant[1]:
                self._is_valid_expression = False
                self._invalid_reason = (
                    f"Pair Rolling operator {qlib_op} has constant '{args[1]}' "
                    f"as its second argument; qlib does not support this"
                )
            return f'{qlib_op}({args[0]}, {args[1]}, {args[2]})', False

        # Unknown operator
        if self.strict_mode:
            raise ValueError(f"Unsupported operator: {op_name}")

        # In non-strict mode, preserve as-is and conservatively treat result as non-constant
        return f'{op_name}({", ".join(args)})', False

    def _convert_operator(self, op_name: str, args: List[str]) -> str:
        """
        Convert a single operator (without const-ness validation; kept for compatibility)

        Args:
            op_name: Operator name
            args: Converted argument list

        Returns:
            Converted expression
        """
        # Cross-sectional operators - need to be implemented in the data processing layer.
        # Return the inner expression directly and mark as needing cross-sectional processing.
        if op_name in self.CROSS_SECTIONAL_OPS:
            self._has_cross_sectional_op = True
            # Rank(x) -> return x directly; cross-sectional Rank is implemented
            # in the data processing layer via CSRankNorm.
            # Note: This means the factor's semantics are "compute the inner expression first,
            # then do cross-sectional ranking in the processing layer"
            return args[0]

        # Unary operators - direct correspondence
        if op_name in self.UNARY_OP_MAP:
            qlib_op = self.UNARY_OP_MAP[op_name]
            return f'{qlib_op}({args[0]})'

        # Unary operators - need expansion
        if op_name in self.UNARY_OP_EXPAND:
            expand_func = self.UNARY_OP_EXPAND[op_name]
            return expand_func(args[0])

        # Binary operators
        if op_name in self.BINARY_OP_MAP:
            qlib_op = self.BINARY_OP_MAP[op_name]
            return f'{qlib_op}({args[0]}, {args[1]})'

        # Rolling operators - direct correspondence
        if op_name in self.ROLLING_OP_MAP:
            qlib_op = self.ROLLING_OP_MAP[op_name]
            return f'{qlib_op}({args[0]}, {args[1]})'

        # Rolling operators - need expansion
        if op_name in self.ROLLING_OP_EXPAND:
            expand_func = self.ROLLING_OP_EXPAND[op_name]
            return expand_func(args[0], args[1])

        # Pair Rolling operators
        if op_name in self.PAIR_ROLLING_OP_MAP:
            qlib_op = self.PAIR_ROLLING_OP_MAP[op_name]
            return f'{qlib_op}({args[0]}, {args[1]}, {args[2]})'

        # Unknown operator
        if self.strict_mode:
            raise ValueError(f"Unsupported operator: {op_name}")

        # In non-strict mode, preserve as-is
        return f'{op_name}({", ".join(args)})'

    def convert_batch(
        self,
        factors: List[Dict[str, Any]],
        keep_original: bool = True,
        filter_invalid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch-convert factor expressions

        Args:
            factors: Factor list in the format [{"expression": "...", "ic": ...}, ...]
            keep_original: Whether to retain the original expression
            filter_invalid: Whether to filter out invalid expressions
                            (e.g., expressions that perform rolling operations on constants)

        Returns:
            Converted factor list containing qlib_expression and needs_cs_rank fields
        """
        result = []
        invalid_count = 0

        for factor in factors:
            if isinstance(factor, dict):
                expr = factor.get('expression', '')
                ic = factor.get('ic', 0.0)
                # Preserve all extra fields from the input factor (except expression and ic, which are processed)
                extra_fields = {k: v for k, v in factor.items()
                               if k not in ['expression', 'ic']}
            else:
                expr = str(factor)
                ic = 0.0
                extra_fields = {}

            try:
                qlib_expr = self.convert(expr)

                # Check if the expression is valid
                if not self._is_valid_expression:
                    invalid_count += 1
                    if filter_invalid:
                        # Filter out invalid expressions; do not add to results
                        continue
                    else:
                        # Do not filter, but mark as invalid
                        converted = {
                            'qlib_expression': None,
                            'original_expression': expr,
                            'ic': ic,
                            'needs_cs_rank': False,
                            'is_valid': False,
                            'invalid_reason': self._invalid_reason
                        }
                        # Preserve extra fields
                        converted.update(extra_fields)
                        result.append(converted)
                        continue

                converted = {
                    'qlib_expression': qlib_expr,
                    'ic': ic,
                    'needs_cs_rank': self._has_cross_sectional_op,
                    'is_valid': True
                }
                if keep_original:
                    converted['original_expression'] = expr
                # Preserve extra fields (such as rank_ic_valid, etc.)
                converted.update(extra_fields)
                result.append(converted)

            except Exception as e:
                print(f"[Warning] Conversion failed: {expr}, error: {e}")
                if not filter_invalid:
                    # Preserve the original expression
                    converted = {
                        'qlib_expression': None,
                        'original_expression': expr,
                        'ic': ic,
                        'needs_cs_rank': False,
                        'is_valid': False,
                        'error': str(e)
                    }
                    # Preserve extra fields
                    converted.update(extra_fields)
                    result.append(converted)

        if invalid_count > 0:
            action = "filtered out" if filter_invalid else "marked"
            print(f"[Converter] {action} {invalid_count} invalid expression(s) (e.g., rolling operations on constants)")

        return result

    def batch_needs_cs_rank(self, converted_factors: List[Dict[str, Any]]) -> bool:
        """
        Check whether any factor in the batch conversion needs cross-sectional Rank processing

        Args:
            converted_factors: Factor list returned by convert_batch

        Returns:
            Whether any factor needs cross-sectional Rank processing
        """
        return any(f.get('needs_cs_rank', False) for f in converted_factors)

    def convert_and_save(
        self,
        input_file: str,
        output_file: str = None,
        format_type: str = 'json'
    ) -> str:
        """
        Read an AlphaSAGE factor file, convert, and save

        Args:
            input_file: Input file path (AlphaSAGE factor JSON)
            output_file: Output file path; defaults to input_file with _qlib suffix
            format_type: Output format ('json' or 'qlib_fields')

        Returns:
            Output file path
        """
        input_path = Path(input_file)

        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        factors = data.get('factors', [])

        # Batch conversion
        converted_factors = self.convert_batch(factors)

        # Determine output path
        if output_file is None:
            output_path = input_path.parent / f"{input_path.stem}_qlib{input_path.suffix}"
        else:
            output_path = Path(output_file)

        # Save results
        if format_type == 'json':
            output_data = {
                'timestamp': data.get('timestamp', ''),
                'factors_count': len(converted_factors),
                'source': str(input_path),
                'factors': converted_factors
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        elif format_type == 'qlib_fields':
            # Output in fields format directly usable by qlib
            fields = []
            names = []
            for i, factor in enumerate(converted_factors):
                if factor.get('qlib_expression'):
                    fields.append(factor['qlib_expression'])
                    ic = factor.get('ic', 0)
                    names.append(f'ALPHA_{i+1:03d}_IC{ic:.4f}')

            output_data = {
                'fields': fields,
                'names': names,
                'factors_detail': converted_factors
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Converter] Conversion complete, saved to: {output_path}")
        print(f"[Converter] Successfully converted {sum(1 for f in converted_factors if f.get('qlib_expression'))} / {len(converted_factors)} factor(s)")

        return str(output_path)


def convert_alphagen_to_qlib(expr: str) -> str:
    """
    Convenience function: Convert an AlphaSAGE expression to a Qlib expression

    Args:
        expr: AlphaSAGE expression

    Returns:
        Qlib expression
    """
    converter = AlphaSAGEToQlibConverter()
    return converter.convert(expr)


def convert_factor_file(input_file: str, output_file: str = None) -> str:
    """
    Convenience function: Convert a factor file

    Args:
        input_file: Input AlphaSAGE factor JSON file
        output_file: Output file path

    Returns:
        Output file path
    """
    converter = AlphaSAGEToQlibConverter()
    return converter.convert_and_save(input_file, output_file)


# Test code
if __name__ == '__main__':
    converter = AlphaSAGEToQlibConverter()

    # Test cases
    test_cases = [
        # Simple features
        ("close", "$close"),
        ("volume", "$volume"),

        # Unary operators
        ("Abs(close)", "Abs($close)"),
        ("Log(close)", "Log($close)"),
        ("SLog1p(close)", "Mul(Sign($close), Log(Add(Abs($close), 1)))"),
        ("Inv(close)", "Div(1, $close)"),

        # Binary operators
        ("Add(close,volume)", "Add($close, $volume)"),
        ("Div(close,volume)", "Div($close, $volume)"),
        ("Pow(close,2)", "Power($close, 2)"),

        # Rolling operators
        ("TsMean(close,20)", "Mean($close, 20)"),
        ("TsStd(volume,30)", "Std($volume, 30)"),
        ("Ref(close,5)", "Ref($close, 5)"),

        # Complex Rolling operators
        ("TsIr(close,20)", "Div(Mean($close, 20), Std($close, 20))"),
        ("TsMinMaxDiff(high,10)", "Sub(Max($high, 10), Min($high, 10))"),

        # Nested expressions
        ("TsIr(TsMean(volume,30),50)", None),  # Complex nesting
        ("TsMin(TsIr(TsMean(volume,30),20),40)", None),  # Deep nesting

        # Pair Rolling
        ("TsCorr(close,volume,20)", "Corr($close, $volume, 20)"),
    ]

    print("=" * 60)
    print("AlphaSAGE -> Qlib Expression Conversion Test")
    print("=" * 60)

    for alphagen_expr, expected in test_cases:
        qlib_expr = converter.convert(alphagen_expr)
        status = "✓" if (expected is None or qlib_expr == expected) else "✗"
        print(f"\n{status} {alphagen_expr}")
        print(f"  -> {qlib_expr}")
        if expected and qlib_expr != expected:
            print(f"  Expected: {expected}")

    # Test invalid expression detection
    print("\n" + "=" * 60)
    print("Invalid Expression Detection Test")
    print("=" * 60)

    invalid_test_cases = [
        # Rolling operations on direct constants (invalid)
        "Sub(TsMed(Constant(-0.01),10), TsMean(volume,10))",  # Med(-0.01, 10)
        "Sub(TsMad(Constant(-5.0),40), volume)",  # Mad(-5.0, 40)
        "TsMean(Constant(1.0),20)",  # Mean(1.0, 20)
        "TsMax(Constant(0),10)",  # Max(0, 10)

        # Rolling operations on nested constant expressions (invalid) - new tests
        "TsWMA(Div(1, 2.0), 50)",  # WMA(Div(1, 2.0), 50) - Div(1,2)=0.5 is constant
        "TsMean(Add(1, 2), 10)",  # Mean(Add(1, 2), 10) - Add(1,2)=3 is constant
        "TsStd(Mul(3, 4), 20)",  # Std(Mul(3, 4), 20) - Mul(3,4)=12 is constant
        "Mul(TsCorr(high, Sub(TsWMA(Div(1, 2.0), 50), volume), 40), TsCorr(high, volume, 20))",  # Complex nesting

        # Constant argument tests for Pair Rolling operators (invalid)
        "TsCorr(Div(1, 2), volume, 20)",  # First argument is constant
        "TsCov(close, Add(1, 1), 30)",  # Second argument is constant

        # Valid expressions
        "TsMean(close,20)",
        "Sub(TsMed(close,10), TsMean(volume,10))",
        "TsWMA(Div(close, volume), 50)",  # Div(close, volume) is not constant
        "Mul(TsCorr(high, Sub(volume, close), 40), TsCorr(high, volume, 20))",  # All non-constant
    ]

    for expr in invalid_test_cases:
        result = converter.convert_with_metadata(expr)
        status = "✓ Valid" if result['is_valid'] else "✗ Invalid"
        print(f"\n{status}: {expr}")
        print(f"  -> {result['qlib_expression']}")
        if not result['is_valid']:
            print(f"  Reason: {result.get('invalid_reason', 'Unknown')}")

    print("\n" + "=" * 60)
    print("Test Complete")
