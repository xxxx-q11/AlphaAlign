"""Factor correlation evaluator"""
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


class CorrelationEvaluator:
    """Factor correlation evaluator - Check correlation between new factor and SOTA factor pool"""
    
    def __init__(self, qlib_path: Optional[str] = None):
        """
        Initialize correlation evaluator
        
        Args:
            qlib_path: Qlib data path, if None then try to auto-detect
        """
        self._qlib_initialized = False
        self._qlib_path = qlib_path
        self._D = None  # Qlib Data API
        self._CSRankNorm = None
        self._preprocess_formula = None
        self._converter = None
    
    def _init_qlib(self) -> bool:
        """Initialize Qlib"""
        if self._qlib_initialized:
            return True
        
        try:
            # Add module path
            modules_path = Path(__file__).parent.parent.parent / "Qlib_MCP" / "modules"
            if str(modules_path) not in sys.path:
                sys.path.insert(0, str(modules_path))
            alphagen_src_path = (
                Path(__file__).parent.parent.parent
                / "Qlib_MCP"
                / "workspace"
                / "AlphaSAGE"
                / "src"
            )
            if str(alphagen_src_path) not in sys.path:
                sys.path.insert(0, str(alphagen_src_path))
            
            from calculate_ic import init_qlib, preprocess_formula
            from alphagen.utils.qlib_converter import AlphaSAGEToQlibConverter
            from qlib.data import D
            from qlib.data.dataset.processor import CSRankNorm
            
            # Initialize Qlib
            if self._qlib_path is not None:
                success, _ = init_qlib(provider_uri=self._qlib_path)
            else:
                success, _ = init_qlib()
                
            if not success:
                print("Warning: Qlib initialization failed")
                return False
            
            self._D = D
            self._CSRankNorm = CSRankNorm
            self._preprocess_formula = preprocess_formula
            self._converter = AlphaSAGEToQlibConverter()
            self._qlib_initialized = True
            return True
            
        except ImportError as e:
            print(f"Warning: Unable to import Qlib module: {e}")
            return False
        except Exception as e:
            print(f"Warning: Qlib initialization exception: {e}")
            return False
    
    def check_correlation(
        self,
        factor: Dict[str, Any],
        sota_factors: List[str],
        instruments: str = "csi300",
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        threshold: float = 0.99
    ) -> bool:
        """
        Check if new factor is highly correlated with SOTA factors
        
        Args:
            factor: New factor dictionary, containing 'qlib_expression' and optional 'needs_cs_rank'
            sota_factors: SOTA factor expression list
            instruments: Stock pool name
            start_date: Start date
            end_date: End date
            threshold: Correlation threshold, >= this value is considered highly correlated
            
        Returns:
            True: Factor is not highly correlated with SOTA factors, can be kept
            False: Factor is highly correlated with SOTA factors, should be removed
        """
        if not factor:
            return True
        
        if not sota_factors:
            print("Warning: SOTA factor pool is empty, skipping correlation check")
            return True
        
        # Initialize Qlib
        if not self._init_qlib():
            print("Warning: Qlib unavailable, skipping correlation check")
            return True
        
        print(f"\n[Checking Factor Correlation] SOTA factor count: {len(sota_factors)}")
        
        try:
            # Get factor expression
            expression = factor.get('qlib_expression')
            if not expression:
                print("Warning: Factor expression is empty")
                return True
            
            needs_cs_rank = factor.get('needs_cs_rank', False)
            
            # Calculate new factor values
            new_factor_values = self._calculate_factor_values(
                expression, instruments, start_date, end_date, needs_cs_rank
            )
            
            if new_factor_values is None:
                return True
            
            # Calculate maximum correlation with SOTA factors
            max_corr = self._calculate_max_correlation(
                new_factor_values, sota_factors, instruments, start_date, end_date
            )
            
            # Judge result
            print(f"Maximum correlation: {max_corr:.4f}, threshold: {threshold}")
            
            if max_corr >= threshold:
                print(f"Factor is highly correlated with SOTA factors, returning False")
                return False
            else:
                print(f"Factor is not highly correlated with SOTA factors, returning True")
                return True
                
        except Exception as e:
            print(f"Warning: Error occurred while calculating correlation: {e}")
            return True
    
    def _calculate_factor_values(
        self,
        expression: str,
        instruments: str,
        start_date: str,
        end_date: str,
        needs_cs_rank: bool = False
    ) -> Optional[pd.Series]:
        """Calculate factor values"""
        factor_values, _ = self._calculate_factor_values_with_error(
            expression=expression,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            needs_cs_rank=needs_cs_rank,
        )
        return factor_values

    def validate_factor_expression(
        self,
        expression: str,
        instruments: str = "csi300",
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        needs_cs_rank: bool = False,
    ) -> Dict[str, Any]:
        """Validate whether a factor expression can be calculated by qlib."""
        if not expression:
            return {
                "is_calculable": False,
                "reason": "qlib_expression is empty, unable to perform qlib computability check",
            }

        if not self._init_qlib():
            return {
                "is_calculable": False,
                "reason": "Qlib initialization failed, unable to verify factor computability",
            }

        factor_values, error = self._calculate_factor_values_with_error(
            expression=expression,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            needs_cs_rank=needs_cs_rank,
        )
        if factor_values is None:
            return {
                "is_calculable": False,
                "reason": error or "Qlib cannot compute this factor expression",
            }

        return {
            "is_calculable": True,
            "reason": "",
            "value_count": int(len(factor_values)),
        }

    def _calculate_factor_values_with_error(
        self,
        expression: str,
        instruments: str,
        start_date: str,
        end_date: str,
        needs_cs_rank: bool = False,
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        """Calculate factor values and return a structured error when it fails."""
        try:
            if self._converter is not None and not self._looks_like_qlib_expression(expression):
                validation_result = self._converter.convert_with_metadata(expression)
                if not validation_result.get("is_valid", True):
                    invalid_reason = validation_result.get("invalid_reason", "unknown reason")
                    print(
                        "Warning: Skipping unsupported qlib expression during factor "
                        f"value calculation: {invalid_reason}. expression={expression}"
                    )
                    return None, str(invalid_reason)

            formula = self._preprocess_formula(expression)
            instrument_list = self._D.list_instruments(
                instruments=self._D.instruments(instruments),
                start_time=start_date,
                end_time=end_date,
                as_list=True
            )
            
            if len(instrument_list) == 0:
                print("Warning: Unable to get stock list")
                return None, "Qlib did not return an available stock list"
            
            factor_df = self._D.features(
                instrument_list,
                [formula],
                start_time=start_date,
                end_time=end_date
            )
            
            if isinstance(factor_df, pd.DataFrame):
                factor_values = factor_df.iloc[:, 0]
            else:
                factor_values = factor_df
            
            if isinstance(factor_values.index, pd.MultiIndex):
                factor_values.index.names = ['datetime', 'instrument']
            
            # CSRank processing
            if needs_cs_rank:
                processor = self._CSRankNorm()
                temp_df = factor_values.to_frame('factor')
                temp_df = processor(temp_df)
                factor_values = temp_df['factor']

            factor_values = factor_values.dropna()
            if factor_values.empty:
                return None, "Qlib computation result is empty or all missing values"
            
            return factor_values, None
            
        except Exception as e:
            print(f"Warning: Failed to calculate factor values: {e}")
            return None, f"Qlib computation failed: {e}"

    @staticmethod
    def _looks_like_qlib_expression(expression: str) -> bool:
        """Detect whether the input is already a Qlib-style expression."""
        if not expression:
            return False
        return bool(re.search(r"\$(close|open|high|low|volume|vwap)\b", expression))

    def _calculate_max_correlation(
        self,
        new_factor_values: pd.Series,
        sota_factors: List[str],
        instruments: str,
        start_date: str,
        end_date: str
    ) -> float:
        """Calculate maximum correlation with SOTA factors"""
        try:
            return self._calculate_matrix_correlation(
                new_factor_values, sota_factors, instruments, start_date, end_date
            )
        except Exception as e:
            print(f"Warning: Matrix calculation failed ({str(e)[:50]}), falling back to sequential calculation")
            return self._calculate_sequential_correlation(
                new_factor_values, sota_factors, instruments, start_date, end_date
            )
    
    def _calculate_matrix_correlation(
        self,
        new_factor_values: pd.Series,
        sota_factors: List[str],
        instruments: str,
        start_date: str,
        end_date: str
    ) -> float:
        """Batch calculate correlation using matrix operations"""
        # Get stock list
        instrument_list = self._D.list_instruments(
            instruments=self._D.instruments(instruments),
            start_time=start_date,
            end_time=end_date,
            as_list=True
        )
        
        # Batch calculate SOTA factor values
        sota_formulas = [self._preprocess_formula(expr) for expr in sota_factors]
        sota_factors_df = self._D.features(
            instrument_list,
            sota_formulas,
            start_time=start_date,
            end_time=end_date
        )
        
        if isinstance(sota_factors_df.index, pd.MultiIndex):
            sota_factors_df.index.names = ['datetime', 'instrument']
        
        # Merge factors
        new_factor_df = new_factor_values.to_frame('new_factor')
        all_factors_df = pd.concat([new_factor_df, sota_factors_df], axis=1).dropna()
        
        if len(all_factors_df) < 10:
            return -np.inf
        
        # Calculate correlation grouped by date
        date_correlations = []
        for date, group in all_factors_df.groupby(level=0):
            if len(group) < 3:
                continue
            
            new_factor_col = group['new_factor'].values.astype(float)
            sota_factor_cols = group.drop(columns=['new_factor']).values.astype(float)
            
            if sota_factor_cols.shape[1] == 0:
                continue
            
            # Normalize
            new_mean = np.mean(new_factor_col)
            new_std = np.std(new_factor_col)
            if new_std < 1e-10:
                continue
            
            new_factor_norm = (new_factor_col - new_mean) / new_std
            
            sota_means = np.mean(sota_factor_cols, axis=0)
            sota_stds = np.std(sota_factor_cols, axis=0)
            valid_mask = sota_stds > 1e-10
            
            if not np.any(valid_mask):
                continue
            
            sota_cols_valid = sota_factor_cols[:, valid_mask]
            sota_means_valid = sota_means[valid_mask]
            sota_stds_valid = sota_stds[valid_mask]
            
            sota_norm = (sota_cols_valid - sota_means_valid) / sota_stds_valid
            correlations = np.mean(new_factor_norm[:, np.newaxis] * sota_norm, axis=0)
            
            full_correlations = np.full(len(sota_factors), np.nan)
            full_correlations[valid_mask] = correlations
            date_correlations.append(full_correlations)
        
        if len(date_correlations) > 0:
            correlation_matrix = np.array(date_correlations)
            avg_correlations = np.nanmean(correlation_matrix, axis=0)
            return np.nanmax(avg_correlations) if len(avg_correlations) > 0 else -np.inf
        
        return -np.inf
    
    def _calculate_sequential_correlation(
        self,
        new_factor_values: pd.Series,
        sota_factors: List[str],
        instruments: str,
        start_date: str,
        end_date: str
    ) -> float:
        """Calculate correlation sequentially (fallback method)"""
        instrument_list = self._D.list_instruments(
            instruments=self._D.instruments(instruments),
            start_time=start_date,
            end_time=end_date,
            as_list=True
        )
        
        max_corr = -np.inf
        
        for sota_expr in sota_factors:
            try:
                sota_formula = self._preprocess_formula(sota_expr)
                sota_factor_df = self._D.features(
                    instrument_list,
                    [sota_formula],
                    start_time=start_date,
                    end_time=end_date
                )
                
                if isinstance(sota_factor_df, pd.DataFrame):
                    sota_factor_values = sota_factor_df.iloc[:, 0]
                else:
                    sota_factor_values = sota_factor_df
                
                if isinstance(sota_factor_values.index, pd.MultiIndex):
                    sota_factor_values.index.names = ['datetime', 'instrument']
                
                aligned = pd.concat(
                    [new_factor_values, sota_factor_values],
                    axis=1,
                    keys=['factor1', 'factor2']
                ).dropna()
                
                if len(aligned) < 10:
                    continue
                
                correlations = []
                for date, group in aligned.groupby(level=0):
                    if len(group) < 3:
                        continue
                    try:
                        corr, _ = pearsonr(group['factor1'], group['factor2'])
                        if not np.isnan(corr):
                            correlations.append(corr)
                    except Exception:
                        continue
                
                if correlations:
                    avg_corr = np.mean(correlations)
                    max_corr = max(max_corr, avg_corr)
                    
            except Exception:
                continue
        
        return max_corr
