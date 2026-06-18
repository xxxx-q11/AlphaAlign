
import os
import argparse
import json
from collections import Counter
import heapq

import numpy as np
import torch

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.random import reseed_everything
from alphagen.utils.qlib_converter import AlphaSAGEToQlibConverter
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from gan.utils.data import get_data_by_year
from datetime import datetime, timezone

QLIB_PATH = '~/.qlib/qlib_data/cn_data'

# ========== Global variables related to experiment result saving ==========
convergence_history = []  # Convergence data for each generation
pool_results_history = []  # Pool results for each generation
experiment_results_dir = ''  # Experiment results save directory
experiment_name = ''  # Experiment name (used to distinguish with/without LLM seeds)

def _metric(x, y, w):
    key = y[0]

    if key in cache:
        return cache[key]
    token_len = key.count('(') + key.count(')')
    if token_len > 20:
        return -1.

    expr = eval(key)
    try:
        factor = expr.evaluate(data)
        factor = normalize_by_day(factor)
        ic = batch_pearsonr(factor, target_factor)
        ic = torch.nan_to_num(ic).mean().item()
    except OutOfDataRangeError:
        ic = -1.
    except Exception:
        # Fallback: any expression structure/data exception should not crash training
        ic = -1.0
    if np.isnan(ic):
        ic = -1.
    cache[key] = ic
    return ic




def try_single():
    top_key = Counter(cache).most_common(1)[0][0]
    try:
        v_valid = eval(top_key).evaluate(data_valid)
        v_test = eval(top_key).evaluate(data_test)
        ic_test = batch_pearsonr(v_test, target_factor_test)
        ic_test = torch.nan_to_num(ic_test,nan=0,posinf=0,neginf=0).mean().item()
        ic_valid = batch_pearsonr(v_valid, target_factor_valid)
        ic_valid = torch.nan_to_num(ic_valid,nan=0,posinf=0,neginf=0).mean().item()
        ric_test = batch_spearmanr(v_test, target_factor_test)
        ric_test = torch.nan_to_num(ric_test,nan=0,posinf=0,neginf=0).mean().item()
        ric_valid = batch_spearmanr(v_valid, target_factor_valid)
        ric_valid = torch.nan_to_num(ric_valid,nan=0,posinf=0,neginf=0).mean().item()
        return {'ic_test': ic_test, 'ic_valid': ic_valid, 'ric_test': ric_test, 'ric_valid': ric_valid}
    except OutOfDataRangeError:
        print ('Out of data range')
        print(top_key)
        exit()
        return {'ic_test': -1., 'ic_valid': -1., 'ric_test': -1., 'ric_valid': -1.}


def try_pool(capacity):
    pool = AlphaPool(capacity=capacity,
                    stock_data=data,
                    target=target,
                    ic_lower_bound=None)

    exprs = []
    for key in dict(Counter(cache).most_common(capacity)):
        exprs.append(eval(key))
    pool.force_load_exprs(exprs)
    pool._optimize(alpha=5e-3, lr=5e-4, n_iter=2000)

    ic_test, ric_test = pool.test_ensemble(data_test, target)
    ic_valid, ric_valid = pool.test_ensemble(data_valid, target)
    return {'ic_test': ic_test, 'ic_valid': ic_valid, 'ric_test': ric_test, 'ric_valid': ric_valid}




def ev():
    global generation, top_factors_heap, max_generations
    global convergence_history, pool_results_history  # Experiment result records
    generation += 1
    
    # Update heap with capacity 200, save top 200 factors with highest IC values
    # Use min heap, heap top is the minimum IC value
    # Build expression to IC value mapping, keep highest IC value (handle same expression in different iterations)
    if generation <= max_generations - 1:
        expr_to_ic = {}
        for expr, ic in cache.items():
            if expr not in expr_to_ic or ic > expr_to_ic[expr]:
                expr_to_ic[expr] = ic
        
        # Get set of expressions already in heap
        existing_exprs = {expr for _, expr in top_factors_heap}
        
        # Update heap: handle new factors or factors with higher IC values
        for expr, ic in expr_to_ic.items():
            if expr in existing_exprs:
                # If expression already exists, check if IC value needs to be updated
                for i, (old_ic, old_expr) in enumerate(top_factors_heap):
                    if old_expr == expr and ic > old_ic:
                        # Update IC value at this position in heap
                        top_factors_heap[i] = (ic, expr)
                        heapq.heapify(top_factors_heap)  # Re-heapify
                        break
            else:
                # Expression doesn't exist, try to add to heap
                if len(top_factors_heap) < 200:
                    heapq.heappush(top_factors_heap, (ic, expr))
                elif ic > top_factors_heap[0][0]:  # If current IC is greater than minimum IC at heap top
                    heapq.heapreplace(top_factors_heap, (ic, expr))
    
    res = (
        [{'pool': 0, 'res': try_single()}] +
        [{'pool': cap, 'res': try_pool(cap)} for cap in (10, 20, 50, 100)]
    )
    print(res)
    
    # ========== Record convergence data for each generation (for ablation experiments) ==========
    # Get statistical metrics for current generation
    top_factors = Counter(cache).most_common(10)
    best_ic_train = top_factors[0][1] if top_factors else 0
    top10_avg_ic = np.mean([ic for _, ic in top_factors]) if top_factors else 0
    
    # Get validation set metrics for best factor
    best_expr = top_factors[0][0] if top_factors else None
    best_valid_metrics = calculate_factor_valid_metrics(best_expr) if best_expr else {}
    
    # Record convergence history
    convergence_record = {
        'generation': generation,
        'best_ic_train': best_ic_train,
        'top10_avg_ic_train': top10_avg_ic,
        'best_ic_valid': best_valid_metrics.get('ic_valid'),
        'best_rank_ic_valid': best_valid_metrics.get('rank_ic_valid'),
        'cache_size': len(cache),
        'positive_ic_count': sum(1 for _, ic in cache.items() if ic > 0),
        'single_best': res[0]['res'] if res else {}  # try_single result
    }
    convergence_history.append(convergence_record)
    
    # Record Pool results
    pool_record = {
        'generation': generation,
        'pools': {}
    }
    for r in res:
        if r['pool'] > 0:
            pool_record['pools'][f'pool_{r["pool"]}'] = r['res']
    pool_results_history.append(pool_record)
    
    global save_dir
    dir_ = save_dir
    #'/path/to/save/results'
    os.makedirs(dir_, exist_ok=True)
    # if generation % 2 == 0:
    #     with open(f'{dir_}/{generation}.json', 'w') as f:
    #         json.dump({'cache': cache, 'res': res}, f)
        
        # Also save qlib format factors
        #save_qlib_factors(dir_, generation, cache)
    
    # Only on the last iteration, merge and save heap and top 100 factors with highest IC values from last generation
    if generation == max_generations:
        try:
            save_qlib_factors(dir_, generation, cache)
            save_top_factors_heap(dir_, generation, cache)
            print(f'[train_GP] All factors saved to directory: {dir_}')
            
            # ========== Save experiment results (for ablation experiment comparison) ==========
            save_experiment_results()
            
        except Exception as e:
            print(f'[train_GP] Error saving factors: {e}')
            import traceback
            traceback.print_exc()


def save_qlib_factors(dir_: str, generation: int, cache: dict, top_n: int = 100):
    """
    Convert mined factors to qlib format and save with standardized train/valid/test metrics.
    
    Args:
        dir_: Save directory
        generation: Current iteration generation number
        cache: Factor cache {expression: IC value}
        top_n: Save top N factors with highest IC values
    """
    # Create converter
    converter = AlphaSAGEToQlibConverter()
    
    # Get top N factors
    top_factors = Counter(cache).most_common(top_n)
    
    # Filter factors with positive IC
    valid_top_factors = [(expr, ic) for expr, ic in top_factors if ic > 0]
    
    factors = build_standard_factor_metrics_batch(valid_top_factors)
    print(f'[train_GP] Calculated standardized metrics for {len(factors)} factors')
    
    # Convert to qlib format, and filter invalid expressions (e.g., expressions doing Rolling operations on constants)
    converted_factors = converter.convert_batch(factors, filter_invalid=True)
    for factor in converted_factors:
        factor.pop('ic', None)
    
    # Build output data
    output_data = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'generation': generation,
        'factors_count': len(converted_factors),
        'source': 'AlphaSAGE_GP',
        'factors': converted_factors,
        # Also save qlib directly usable fields format (only contains valid expressions)
        'qlib_fields': [f['qlib_expression'] for f in converted_factors if f.get('qlib_expression') and f.get('is_valid', True)],
        'qlib_names': [f'ALPHA_{i+1:03d}' for i, f in enumerate(converted_factors) if f.get('qlib_expression') and f.get('is_valid', True)]
    }
    
    # Save qlib format factors
    qlib_output_file = f'{dir_}/{generation}_qlib_factors.json'
    with open(qlib_output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f'[train_GP] Qlib format factors saved: {qlib_output_file}')
    print(f'[train_GP] Successfully converted {len(output_data["qlib_fields"])} factors')


def calculate_rank_ic_for_factor(expr_str: str):
    """
    Calculate rank_ic value for a single factor on validation set
    
    Args:
        expr_str: Factor expression string
        
    Returns:
        float: rank_ic value on validation set, returns None if calculation fails
    """
    try:
        expr = eval(expr_str)
        v_valid = expr.evaluate(data_valid)
        v_valid = normalize_by_day(v_valid)
        ric_valid = batch_spearmanr(v_valid, target_factor_valid)
        ric_valid = torch.nan_to_num(ric_valid, nan=0, posinf=0, neginf=0).mean().item()
        return ric_valid
    except (OutOfDataRangeError, Exception) as e:
        print(f'[train_GP] Failed to calculate rank_ic: {expr_str}, error: {str(e)}')
        return None


def calculate_rank_ic_batch(expr_list: list):
    """
    Batch calculate rank_ic values for multiple factors on validation set
    
    Args:
        expr_list: Factor expression string list
        
    Returns:
        list: rank_ic value for each factor, returns None if calculation fails
    """
    print(f'[train_GP] Starting batch calculation of rank_ic for {len(expr_list)} factors...')
    
    # Batch evaluate all factors
    valid_factors = []
    valid_indices = []
    
    for i, expr_str in enumerate(expr_list):
        try:
            expr = eval(expr_str)
            v_valid = expr.evaluate(data_valid)
            v_valid = normalize_by_day(v_valid)
            valid_factors.append(v_valid)
            valid_indices.append(i)
        except (OutOfDataRangeError, Exception) as e:
            print(f'[train_GP] Factor {i} evaluation failed: {expr_str[:50]}..., error: {str(e)}')
    
    # Initialize result list
    rank_ic_list = [None] * len(expr_list)
    
    if len(valid_factors) == 0:
        print(f'[train_GP] Warning: No valid factors to calculate rank_ic')
        return rank_ic_list
    
    print(f'[train_GP] Successfully evaluated {len(valid_factors)}/{len(expr_list)} factors, starting batch rank_ic calculation...')
    
    # Batch calculate rank_ic for all valid factors
    try:
        # Calculate rank_ic (Spearman correlation coefficient) with target for each factor
        rank_ics = []
        for i, factor_tensor in enumerate(valid_factors):
            ric = batch_spearmanr(factor_tensor, target_factor_valid)
            ric = torch.nan_to_num(ric, nan=0, posinf=0, neginf=0).mean().item()
            rank_ics.append(ric)
            
            # Print progress every 100 factors
            if (i + 1) % 100 == 0:
                print(f'[train_GP] Calculated rank_ic for {i+1}/{len(valid_factors)} factors')
        
        # Fill results to corresponding positions
        for idx, ric_val in zip(valid_indices, rank_ics):
            rank_ic_list[idx] = ric_val
        
        print(f'[train_GP] Batch calculation completed, successfully calculated {len(rank_ics)} rank_ic values')
        
    except Exception as e:
        print(f'[train_GP] Error in batch rank_ic calculation: {str(e)}, falling back to individual calculation')
        # If batch calculation fails, fall back to individual calculation
        for idx in valid_indices:
            try:
                expr = eval(expr_list[idx])
                v_valid = expr.evaluate(data_valid)
                v_valid = normalize_by_day(v_valid)
                ric = batch_spearmanr(v_valid, target_factor_valid)
                ric = torch.nan_to_num(ric, nan=0, posinf=0, neginf=0).mean().item()
                rank_ic_list[idx] = ric
            except Exception as e2:
                print(f'[train_GP] Factor {idx} calculation failed: {str(e2)}')
    
    return rank_ic_list


def calculate_split_metrics_batch(
    expr_list: list,
    split_data,
    split_target_factor,
    split_name: str,
):
    """
    Batch calculate IC and Rank IC values for multiple factors on a given data split.
    """
    print(f'[train_GP] Starting batch calculation of {split_name} IC/Rank IC for {len(expr_list)} factors...')

    metrics_list = []
    for i, expr_str in enumerate(expr_list):
        try:
            expr = eval(expr_str)
            factor_tensor = expr.evaluate(split_data)
            factor_tensor = normalize_by_day(factor_tensor)

            ic_value = batch_pearsonr(factor_tensor, split_target_factor)
            ic_value = torch.nan_to_num(ic_value, nan=0, posinf=0, neginf=0).mean().item()

            rank_ic_value = batch_spearmanr(factor_tensor, split_target_factor)
            rank_ic_value = torch.nan_to_num(rank_ic_value, nan=0, posinf=0, neginf=0).mean().item()

            metrics_list.append({
                'ic': ic_value,
                'rank_ic': rank_ic_value,
            })
        except (OutOfDataRangeError, Exception) as e:
            print(f'[train_GP] {split_name} metrics calculation failed for factor {i}: {expr_str[:50]}..., error: {e}')
            metrics_list.append({
                'ic': None,
                'rank_ic': None,
            })

        if (i + 1) % 50 == 0 or (i + 1) == len(expr_list):
            print(f'[train_GP] {split_name} metrics calculated for {i+1}/{len(expr_list)} factors')

    return metrics_list


def build_standard_factor_metrics_batch(factors_with_train_ic: list):
    """
    Build standardized train/valid/test metric payloads for factor result files.
    """
    if not factors_with_train_ic:
        return []

    expr_list = [expr for expr, _ in factors_with_train_ic]
    train_metrics_list = calculate_split_metrics_batch(expr_list, data, target_factor, 'train')
    valid_metrics_list = calculate_split_metrics_batch(expr_list, data_valid, target_factor_valid, 'valid')
    test_metrics_list = calculate_split_metrics_batch(expr_list, data_test, target_factor_test, 'test')

    standardized_factors = []
    for (expr, cached_train_ic), train_metrics, valid_metrics, test_metrics in zip(
        factors_with_train_ic,
        train_metrics_list,
        valid_metrics_list,
        test_metrics_list,
    ):
        train_ic = cached_train_ic if cached_train_ic is not None else train_metrics.get('ic')
        train_rank_ic = train_metrics.get('rank_ic')
        valid_ic = valid_metrics.get('ic')
        valid_rank_ic = valid_metrics.get('rank_ic')
        test_ic = test_metrics.get('ic')
        test_rank_ic = test_metrics.get('rank_ic')

        standardized_factors.append({
            'expression': expr,
            'train_ic': train_ic,
            'train_rank_ic': train_rank_ic,
            'valid_ic': valid_ic,
            'valid_rank_ic': valid_rank_ic,
            'test_ic': test_ic,
            'test_rank_ic': test_rank_ic,
            'metrics': {
                'train': {
                    'ic': train_ic,
                    'rank_ic': train_rank_ic,
                    'icir': 0.0,
                    'rank_icir': 0.0,
                },
                'valid': {
                    'ic': valid_ic,
                    'rank_ic': valid_rank_ic,
                    'icir': 0.0,
                    'rank_icir': 0.0,
                },
                'test': {
                    'ic': test_ic,
                    'rank_ic': test_rank_ic,
                    'icir': 0.0,
                    'rank_icir': 0.0,
                },
            },
        })

    return standardized_factors


def calculate_factor_test_metrics(expr_str: str):
    """
    Calculate IC and Rank IC for a single factor on test set
    
    Args:
        expr_str: Factor expression string
        
    Returns:
        dict: {'ic_test': float, 'rank_ic_test': float}, returns None values if calculation fails
    """
    try:
        expr = eval(expr_str)
        v_test = expr.evaluate(data_test)
        v_test = normalize_by_day(v_test)

        ic_test = batch_pearsonr(v_test, target_factor_test)
        ic_test = torch.nan_to_num(ic_test, nan=0, posinf=0, neginf=0).mean().item()

        ric_test = batch_spearmanr(v_test, target_factor_test)
        ric_test = torch.nan_to_num(ric_test, nan=0, posinf=0, neginf=0).mean().item()

        return {'ic_test': ic_test, 'rank_ic_test': ric_test}
    except Exception as e:
        return {'ic_test': None, 'rank_ic_test': None}


def calculate_factor_valid_metrics(expr_str: str):
    """
    Calculate IC and Rank IC for a single factor on validation set
    
    Args:
        expr_str: Factor expression string
        
    Returns:
        dict: {'ic_valid': float, 'rank_ic_valid': float}, returns None values if calculation fails
    """
    try:
        expr = eval(expr_str)
        v_valid = expr.evaluate(data_valid)
        v_valid = normalize_by_day(v_valid)

        ic_valid = batch_pearsonr(v_valid, target_factor_valid)
        ic_valid = torch.nan_to_num(ic_valid, nan=0, posinf=0, neginf=0).mean().item()

        ric_valid = batch_spearmanr(v_valid, target_factor_valid)
        ric_valid = torch.nan_to_num(ric_valid, nan=0, posinf=0, neginf=0).mean().item()

        return {'ic_valid': ic_valid, 'rank_ic_valid': ric_valid}
    except Exception as e:
        return {'ic_valid': None, 'rank_ic_valid': None}


def calculate_correlation_matrix(expr_list: list, top_n: int = 50):
    """
    Calculate correlation matrix for top N factors
    
    Args:
        expr_list: List of factor expression strings
        top_n: Number of top factors to calculate correlation for
        
    Returns:
        dict: Correlation matrix data
    """
    global data_valid
    
    # Limit to top N factors
    expr_list = expr_list[:top_n]
    
    if len(expr_list) == 0:
        return {
            'expressions': [],
            'correlation_matrix': [],
            'top_n': top_n
        }
    
    # Evaluate all factors on validation set
    factors_tensors = []
    valid_exprs = []
    
    for expr_str in expr_list:
        try:
            expr = eval(expr_str)
            factor = expr.evaluate(data_valid)
            factor = normalize_by_day(factor)
            factors_tensors.append(factor)
            valid_exprs.append(expr_str)
        except Exception as e:
            print(f'[train_GP] Failed to evaluate factor for correlation: {expr_str[:50]}..., error: {e}')
            continue
    
    if len(factors_tensors) == 0:
        return {
            'expressions': [],
            'correlation_matrix': [],
            'top_n': top_n
        }
    
    # Calculate pairwise correlation matrix
    n = len(factors_tensors)
    correlation_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                try:
                    # Calculate Pearson correlation
                    corr = batch_pearsonr(factors_tensors[i], factors_tensors[j])
                    corr = torch.nan_to_num(corr, nan=0, posinf=0, neginf=0).mean().item()
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
                except Exception as e:
                    print(f'[train_GP] Failed to calculate correlation between factor {i} and {j}: {e}')
                    correlation_matrix[i, j] = 0.0
                    correlation_matrix[j, i] = 0.0
    
    return {
        'expressions': valid_exprs,
        'correlation_matrix': correlation_matrix.tolist(),
        'top_n': len(valid_exprs)
    }


def save_experiment_results():
    """
    Save experiment results for ablation experiment comparison
    """
    global cache, generation, max_generations, experiment_results_dir, experiment_name
    global convergence_history, pool_results_history
    
    os.makedirs(experiment_results_dir, exist_ok=True)
    print(f'[train_GP] Starting to save experiment results to: {experiment_results_dir}')
    
    # ========== 1. Save convergence history (for line chart) ==========
    convergence_file = os.path.join(experiment_results_dir, 'convergence_history.json')
    with open(convergence_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': experiment_name,
            'total_generations': max_generations,
            'history': convergence_history
        }, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Convergence history saved: {convergence_file}')
    
    # ========== 2. Save final factor details (for box plot) ==========
    # Get Top-200 factors
    top_factors = Counter(cache).most_common(200)
    final_factors = []
    
    print(f'[train_GP] Calculating complete metrics for Top-200 factors...')
    for i, (expr, ic_train) in enumerate(top_factors):
        if ic_train <= 0:
            continue
        
        factor_info = {
            'rank': i + 1,
            'expression': expr,
            'ic_train': ic_train
        }
        
        # Calculate validation set metrics
        valid_metrics = calculate_factor_valid_metrics(expr)
        factor_info.update(valid_metrics)
        
        # Calculate test set metrics
        test_metrics = calculate_factor_test_metrics(expr)
        factor_info.update(test_metrics)
        
        final_factors.append(factor_info)
        
        if (i + 1) % 50 == 0:
            print(f'[train_GP] Calculated metrics for {i+1}/{len(top_factors)} factors')
    
    final_factors_file = os.path.join(experiment_results_dir, 'final_factors.json')
    with open(final_factors_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': experiment_name,
            'total_factors': len(final_factors),
            'factors': final_factors
        }, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Final factor details saved: {final_factors_file}')
    
    # ========== 3. Save Pool results (for bar chart) ==========
    # Get Pool results from last generation
    if pool_results_history:
        final_pool_results = pool_results_history[-1] if pool_results_history else {}
    else:
        final_pool_results = {}
    
    pool_results_file = os.path.join(experiment_results_dir, 'pool_results.json')
    with open(pool_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': experiment_name,
            'final_generation': generation,
            'pool_results': final_pool_results,
            'pool_history': pool_results_history
        }, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Pool results saved: {pool_results_file}')
    
    # ========== 4. Save factor correlation matrix (for heatmap) ==========
    expr_list = [expr for expr, ic in top_factors if ic > 0]
    corr_data = calculate_correlation_matrix(expr_list, top_n=50)
    
    correlation_file = os.path.join(experiment_results_dir, 'correlation_matrix.json')
    with open(correlation_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_name': experiment_name,
            **corr_data
        }, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Correlation matrix saved: {correlation_file}')
    
    # ========== 5. Save experiment summary (for main results table) ==========
    # Calculate summary metrics
    valid_factors = [f for f in final_factors if f.get('ic_train') and f['ic_train'] > 0]
    
    summary = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'total_generations': max_generations,
        'final_generation': generation,
        
        # Factor count statistics
        'total_cache_size': len(cache),
        'valid_factors_count': len(valid_factors),
        'positive_ic_count': sum(1 for _, ic in cache.items() if ic > 0),
        
        # Best factor metrics
        'best_ic_train': max([f['ic_train'] for f in valid_factors]) if valid_factors else None,
        'best_ic_valid': max([f['ic_valid'] for f in valid_factors if f.get('ic_valid')]) if valid_factors else None,
        'best_ic_test': max([f['ic_test'] for f in valid_factors if f.get('ic_test')]) if valid_factors else None,
        'best_rank_ic_valid': max([f['rank_ic_valid'] for f in valid_factors if f.get('rank_ic_valid')]) if valid_factors else None,
        'best_rank_ic_test': max([f['rank_ic_test'] for f in valid_factors if f.get('rank_ic_test')]) if valid_factors else None,
        
        # Top-10 factor average metrics
        'top10_avg_ic_train': np.mean([f['ic_train'] for f in valid_factors[:10]]) if len(valid_factors) >= 10 else None,
        'top10_avg_ic_valid': np.mean([f['ic_valid'] for f in valid_factors[:10] if f.get('ic_valid')]) if len(valid_factors) >= 10 else None,
        'top10_avg_ic_test': np.mean([f['ic_test'] for f in valid_factors[:10] if f.get('ic_test')]) if len(valid_factors) >= 10 else None,
        
        # Pool combination effect
        'pool_results': final_pool_results,
        
        # Convergence information
        'convergence_generation': None,  # Generation when threshold first reached
    }
    
    # Calculate convergence generation (first generation where best_ic_train > 0.03)
    ic_threshold = 0.03
    for record in convergence_history:
        if record.get('best_ic_train', 0) > ic_threshold:
            summary['convergence_generation'] = record['generation']
            break
    
    summary_file = os.path.join(experiment_results_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[train_GP] Experiment summary saved: {summary_file}')
    
    print(f'[train_GP] ========== Experiment results saving completed ==========')


def save_top_factors_heap(dir_: str, generation: int, cache: dict):
    """
    On the last iteration, merge and save factors from heap and top 100 factors by IC value
    from current cache, and convert to qlib format with standardized split metrics.
    
    Args:
        dir_: Save directory
        generation: Current iteration generation number
        cache: Factor cache {expression: IC value}
    """
    global top_factors_heap
    
    converter = AlphaSAGEToQlibConverter()

    # 1. Extract all factors from heap and sort by IC value descending
    heap_factors = list(top_factors_heap)
    heap_factors.sort(reverse=True)  # Sort by IC value descending

    # 2. Get top 100 factors by IC value from current cache
    top_cache_factors = Counter(cache).most_common(100)

    # 3. Merge factors from heap and cache top-100, deduplicate and keep highest IC value
    merged_factors_dict = {}

    # Add factors from heap
    for ic, expr in heap_factors:
        if ic > 0:  # Only keep factors with positive IC
            if expr not in merged_factors_dict or ic > merged_factors_dict[expr]:
                merged_factors_dict[expr] = ic

    # Add top-100 factors from cache
    for expr, ic in top_cache_factors:
        if ic > 0:  # Only keep factors with positive IC
            if expr not in merged_factors_dict or ic > merged_factors_dict[expr]:
                merged_factors_dict[expr] = ic

    # Convert to factor list format, sort by IC value descending, and fill in train/valid/test metrics
    sorted_factors = sorted(merged_factors_dict.items(), key=lambda x: x[1], reverse=True)
    factors = build_standard_factor_metrics_batch(sorted_factors)

    # Convert to qlib format, and filter invalid expressions (e.g., Rolling operations on constants)
    converted_factors = converter.convert_batch(factors, filter_invalid=True)
    for factor in converted_factors:
        factor.pop('ic', None)

    # Build output data
    output_data = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'generation': generation,
        'factors_count': len(converted_factors),
        'source': 'AlphaSAGE_GP_Heap_Merged',
        'heap_factors_count': len(heap_factors),
        'cache_top100_count': len(top_cache_factors),
        'merged_unique_count': len(merged_factors_dict),
        'factors': converted_factors,
        # Also save qlib directly usable fields format (only contains valid expressions)
        'qlib_fields': [f['qlib_expression'] for f in converted_factors if f.get('qlib_expression') and f.get('is_valid', True)],
        'qlib_names': [f'ALPHA_{i+1:03d}' for i, f in enumerate(converted_factors) if f.get('qlib_expression') and f.get('is_valid', True)]
    }

    # Save merged factors
    heap_output_file = f'{dir_}/{generation}_top200_factors_heap_merged.json'
    with open(heap_output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f'[train_GP] Merged factors saved: {heap_output_file}')
    print(f'[train_GP] Heap factor count: {len(heap_factors)}, Cache top-100 factor count: {len(top_cache_factors)}, After merge & dedup: {len(merged_factors_dict)}, Successfully converted: {len(output_data["qlib_fields"])} factors')





def run(args):
    if args.instruments == 'sp500' or args.instruments == 'nasdaq100':
        QLIB_PATH = '~/.qlib/qlib_data/us_data'
    else:
        QLIB_PATH = '~/.qlib/qlib_data/cn_data'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    reseed_everything(args.seed)

    global data, data_valid, data_test, target, target_factor, target_factor_valid, target_factor_test, cache, generation, save_dir, top_factors_heap, max_generations
    # Global variables related to experiment result saving
    global convergence_history, pool_results_history, experiment_results_dir, experiment_name

    close = Feature(FeatureType.CLOSE)
    target_horizon_days = max(int(getattr(args, 'target_horizon_days', 10)), 1)
    target = Ref(close, -(target_horizon_days + 1)) / Ref(close, -1) - 1
    print(
        f'[train_GP] Target horizon: T+{target_horizon_days + 1}/T+1 - 1 '
        f'({target_horizon_days} trading-day holding return)'
    )

    train_start_time = '2010-01-01'
    train_end_time = f'{args.train_end_year}-12-31'
    valid_start_time = f'{args.train_end_year + 1}-01-01'
    valid_end_time = f'{args.train_end_year + 1}-12-31'
    test_start_time = f'{args.train_end_year + 2}-01-01'
    test_end_time = f'{args.train_end_year + 4}-12-31'

    data = StockData(instrument=args.instruments,
                           start_time=train_start_time,
                           end_time=train_end_time,
                           qlib_path=QLIB_PATH)
    data_valid = StockData(instrument=args.instruments,
                           start_time=valid_start_time,
                           end_time=valid_end_time,
                           qlib_path=QLIB_PATH)
    data_test = StockData(instrument=args.instruments,
                          start_time=test_start_time,
                          end_time=test_end_time,
                          qlib_path=QLIB_PATH)
                          
    save_dir = f'data/gp_{args.instruments}_{args.train_end_year}_{args.freq}_{args.seed}_{datetime.now(timezone.utc).strftime("%m%d_%H%M")}'                      
    #save_dir = f'data/gp_{args.instruments}_{args.train_end_year}_{args.freq}_{args.seed}_{datetime.now().strftime("%m%d%H%M")}'
    #save_dir = f'data/{args.instruments}_{args.train_end_year}_{args.freq}_{args.seed}'

    Metric = make_fitness(function=_metric, greater_is_better=True)
    funcs = [make_function(**func._asdict()) for func in generic_funcs]

    generation = 0
    cache = {}
    top_factors_heap = []  # Min-heap with capacity 200, stores top 200 factors with highest IC values

    # ========== Initialize experiment result tracking variables ==========
    convergence_history = []
    pool_results_history = []

    target_factor = target.evaluate(data)
    target_factor_valid = target.evaluate(data_valid)
    target_factor_test = target.evaluate(data_test)

    # ========== Load seed factors (from mining_feedback["suggested_seeds"]) ==========
    seed_factors = []
    seed_factors_original = []  # Save original seed factors
    seed_factors_file = getattr(args, 'seed_factors_file', '')

    # Set experiment results save directory and experiment name

    #experiment_results_dir = f'data/experiment_results_{args.seed}'
    experiment_results_dir = f'AlphaSAGE/data/experiment_results_1_No_LLM_seed'
    # Set experiment name based on whether LLM seed factors are used
    if seed_factors_file:
        experiment_name = f'with_LLM_seed_{args.instruments}_{args.train_end_year}_seed{args.seed}'
    else:
        experiment_name = f'without_LLM_seed_{args.instruments}_{args.train_end_year}_seed{args.seed}'

    print(f'[train_GP] Experiment name: {experiment_name}')
    print(f'[train_GP] Experiment results will be saved to: {experiment_results_dir}')

    if seed_factors_file:
        seed_factors_original = load_seed_factors(seed_factors_file)
        if seed_factors_original:
            # Amplify seed factors: target ~300 to more effectively guide GP
            # Amplification ratio ~50% of initial population (1000 * 0.5 = 500, but targeting 300 to account for validation failures)
            amplify_target = min(400, int(1000 * 0.5))  # Target amplification count
            seed_factors = amplify_seed_factors(
                seed_factors_original,
                target_count=amplify_target,
                random_state=np.random.RandomState(args.seed)
            )

            # Pre-inject all seed factors (original + amplified) into cache
            inject_seed_factors_to_cache(seed_factors, data, target_factor)
            print(f'[train_GP] Seed factors loaded: original={len(seed_factors_original)}, after amplification={len(seed_factors)}, will be added to initial population')

    # ========== Build terminals (base features + constants, no longer include seed factors) ==========
    features = ['open_', 'close', 'high', 'low', 'volume', 'vwap']
    constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
    
    terminals = features + constants
    print(f'[train_GP] Total terminals: {len(terminals)} (base features: {len(features)}, constants: {len(constants)})')

    X_train = np.array([terminals])
    y_train = np.array([[1]])

    max_generations = 30  # Total number of iterations
    est_gp = SymbolicRegressor(population_size=1000,
                            generations=max_generations,
                            init_depth=(2, 6),
                            tournament_size=600,
                            stopping_criteria=1.,
                            p_crossover=0.3,
                            p_subtree_mutation=0.1,
                            p_hoist_mutation=0.01,
                            p_point_mutation=0.1,
                            p_point_replace=0.6,
                            max_samples=0.9,
                            verbose=1,
                            parsimony_coefficient=0.,
                            random_state=args.seed,
                            function_set=funcs,
                            metric=Metric,
                            const_range=None,
                            n_jobs=1)
    
    # If seed factors exist, use warm_start: first fit once to generate initial population, then inject seed factors, then continue fitting
    if seed_factors:
            # Step 1: Only generate initial population (generations=1)
            est_gp.set_params(generations=1, warm_start=False)
            est_gp.fit(X_train, y_train, callback=None)  # First fit without callback to avoid redundant computation

            # Inject seed factors into initial population (replace some individuals with lower fitness)
            actually_injected = inject_seed_factors_to_initial_population(
                est_gp, seed_factors, X_train, y_train, terminals, len(features),
                replace_ratio=0.5  # Replace 50% of initial population so seed factors guide evolution more effectively
            )
            print(f'[train_GP] Successfully injected {actually_injected}/{len(seed_factors)} seed factors into initial population')

            # Step 2: Use warm_start to continue evolving remaining generations
            # Important: set generation to 1 since the first fit already completed 1 generation
            generation = 1
            est_gp.set_params(generations=max_generations, warm_start=True)
            est_gp.fit(X_train, y_train, callback=ev)
    else:
        est_gp.fit(X_train, y_train, callback=ev)
    
    print(est_gp._program.execute(X_train))


def has_future_leakage(expr_str: str) -> bool:
    """
    Check whether an expression contains future information leakage.

    Future information leakage cases:
    - Ref(x, -n) where n > 0, representing data from n days in the future

    Args:
        expr_str: Expression string

    Returns:
        bool: True if future information leakage is detected
    """
    import re
    # Match Ref(xxx, -number) pattern, where negative number indicates future data
    ref_pattern = r'Ref\([^,]+,\s*-(\d+)\)'
    matches = re.findall(ref_pattern, expr_str)
    if matches:
        for val in matches:
            if int(val) > 0:  # e.g. Ref(x, -5) where 5 > 0
                return True
    return False


def load_seed_factors(seed_factors_file: str) -> list:
    """
    Load seed factors from file and validate them.
    Also check for and filter out factors with future information leakage.

    Args:
        seed_factors_file: Path to seed factor JSON file

    Returns:
        list: List of valid seed factor expressions (without future information leakage)
    """
    if not seed_factors_file or not os.path.exists(seed_factors_file):
        print(f'[train_GP] Seed factor file does not exist: {seed_factors_file}')
        return []
    
    try:
        with open(seed_factors_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        seed_factors_raw = data.get('seed_factors', [])
        print(f'[train_GP] Loaded {len(seed_factors_raw)} seed factors from file')

        # Validate and filter valid seed factors
        valid_seeds = []
        leakage_count = 0
        for sf in seed_factors_raw:
            expr_str = sf.get('expression', '') if isinstance(sf, dict) else str(sf)
            if not expr_str:
                continue
            
            # Check for future information leakage
            if has_future_leakage(expr_str):
                print(f'[train_GP] Seed factor has future information leakage, filtered out: {expr_str[:60]}...')
                leakage_count += 1
                continue

            # Try parsing the expression to verify validity
            try:
                _ = eval(expr_str)
                valid_seeds.append(expr_str)
                print(f'[train_GP] Seed factor valid: {expr_str[:60]}...')
            except Exception as e:
                print(f'[train_GP] Seed factor invalid: {expr_str[:60]}..., error: {e}')

        print(f'[train_GP] Valid seed factors after verification: {len(valid_seeds)}/{len(seed_factors_raw)} (filtered future leakage: {leakage_count})')
        return valid_seeds

    except Exception as e:
        print(f'[train_GP] Failed to load seed factor file: {e}')
        return []


def amplify_seed_factors(seed_factors: list, target_count: int = 200, random_state=None) -> list:
    """
    Amplify seed factors by generating more variants through mutation.

    Amplification strategies:
    1. Parameter mutation: change time window parameters (10, 20, 30, 40, 50)
    2. Operator replacement: replace with similar operators (TsMean <-> TsEMA, Add <-> Sub, etc.)
    3. Feature replacement: replace base features (close <-> vwap, high <-> low, etc.)
    4. Sub-expression crossover: combine sub-expressions from different seed factors

    Args:
        seed_factors: List of original seed factor expression strings
        target_count: Target amplification count
        random_state: Random number generator

    Returns:
        list: Amplified seed factor list (including original factors)
    """
    import re
    import random
    
    if random_state is None:
        random_state = random.Random(42)
    elif isinstance(random_state, np.random.RandomState):
        # If it is a numpy RandomState, convert to Python random
        random_state = random.Random(random_state.randint(0, 2**31))
    
    if not seed_factors:
        return []
    
    # Time window parameter set
    time_windows = [5, 10, 15, 20, 30, 40, 50, 60]

    # Replaceable operator mappings (similar functionality)
    operator_replacements = {
        'TsMean': ['TsEMA', 'TsWMA', 'TsMed'],
        'TsEMA': ['TsMean', 'TsWMA'],
        'TsWMA': ['TsMean', 'TsEMA'],
        'TsMed': ['TsMean'],
        'TsStd': ['TsMad', 'TsVar'],
        'TsMad': ['TsStd'],
        'TsVar': ['TsStd'],
        'TsMax': ['TsMin', 'TsArgMax'],
        'TsMin': ['TsMax', 'TsArgMin'],
        'TsRank': ['TsArgMax', 'TsArgMin'],
        'Add': ['Sub'],
        'Sub': ['Add'],
        'Mul': ['Div'],
        'Div': ['Mul'],
        'Greater': ['Less'],
        'Less': ['Greater'],
    }
    
    # Replaceable feature mappings
    feature_replacements = {
        'close': ['vwap', 'open_', 'high', 'low'],
        'vwap': ['close', 'open_'],
        'open_': ['close', 'vwap'],
        'high': ['low', 'close'],
        'low': ['high', 'close'],
        'volume': ['close', 'vwap'],
    }
    
    # Store all variants (including original factors)
    all_variants = set(seed_factors)
    
    def mutate_time_window(expr_str: str) -> list:
        """Mutate time-window parameters.
        
        Note: to avoid future-information leakage, all time-window parameters use positive values.
        - Ts operators: positive values indicate lookback window length.
        - Ref operator: positive values indicate offsets into the past
                        (for example, Ref(close, 5) = close from 5 days ago).
                        Negative values indicate offsets into the future
                        (information leakage and should be avoided).
        """
        variants = []
        # Match patterns like TsXxx(..., number) or Ref(..., number).
        pattern = r'(Ts\w+|Ref)\(([^,]+),\s*(-?\d+)\)'
        matches = list(re.finditer(pattern, expr_str))
        
        if not matches:
            return variants
        
        for match in matches:
            op, args, window = match.groups()
            window_int = int(window)
            
            # Skip negative Ref parameters because they may cause future-information leakage.
            if op == 'Ref' and window_int < 0:
                # Convert negative Ref values to positive values to fix potential issues
                # in the original expression; Ref(x, -5) becomes Ref(x, 5).
                continue
            
            # Try different time windows, all positive to avoid future-information leakage.
            for new_window in time_windows:
                if new_window != abs(window_int):
                    # All operators use positive time windows.
                    new_expr = expr_str[:match.start()] + f'{op}({args}, {new_window})' + expr_str[match.end():]
                    variants.append(new_expr)
        
        return variants
    
    def mutate_operator(expr_str: str) -> list:
        """Replace operators."""
        variants = []
        for op, replacements in operator_replacements.items():
            if op in expr_str:
                for new_op in replacements:
                    new_expr = expr_str.replace(op, new_op, 1)  # Replace only the first occurrence.
                    if new_expr != expr_str:
                        variants.append(new_expr)
        return variants
    
    def mutate_feature(expr_str: str) -> list:
        """Replace base features."""
        variants = []
        for feat, replacements in feature_replacements.items():
            # Match standalone feature names, not parts of operators.
            pattern = rf'\b{feat}\b'
            if re.search(pattern, expr_str):
                for new_feat in replacements:
                    new_expr = re.sub(pattern, new_feat, expr_str, count=1)
                    if new_expr != expr_str:
                        variants.append(new_expr)
        return variants
    
    def is_valid_expr(expr_str: str) -> bool:
        """Validate the expression and check for future-information leakage."""
        try:
            # 1. Check future-information leakage: negative Ref parameters.
            # Ref(x, -n) indicates data n days in the future, which is leakage.
            ref_pattern = r'Ref\([^,]+,\s*(-\d+)\)'
            ref_matches = re.findall(ref_pattern, expr_str)
            for ref_val in ref_matches:
                if int(ref_val) < 0:
                    return False  # Future-information leakage exists; reject this expression.
            
            # 2. Validate expression syntax.
            expr = eval(expr_str)
            return expr is not None
        except Exception:
            return False
    
    print(f'[train_GP] Starting seed factor augmentation: original_count={len(seed_factors)}, target_count={target_count}')
    
    # Iteratively augment until the target count is reached.
    iteration = 0
    max_iterations = 20  # Prevent infinite loops.
    
    while len(all_variants) < target_count and iteration < max_iterations:
        iteration += 1
        current_factors = list(all_variants)
        new_variants = []
        
        for expr_str in current_factors:
            if len(all_variants) + len(new_variants) >= target_count:
                break
            
            # 1. Time-window mutation.
            window_variants = mutate_time_window(expr_str)
            for v in window_variants:
                if v not in all_variants and is_valid_expr(v):
                    new_variants.append(v)
                    if len(all_variants) + len(new_variants) >= target_count:
                        break
            
            # 2. Operator mutation.
            if len(all_variants) + len(new_variants) < target_count:
                op_variants = mutate_operator(expr_str)
                for v in op_variants:
                    if v not in all_variants and v not in new_variants and is_valid_expr(v):
                        new_variants.append(v)
                        if len(all_variants) + len(new_variants) >= target_count:
                            break
            
            # 3. Feature mutation.
            if len(all_variants) + len(new_variants) < target_count:
                feat_variants = mutate_feature(expr_str)
                for v in feat_variants:
                    if v not in all_variants and v not in new_variants and is_valid_expr(v):
                        new_variants.append(v)
                        if len(all_variants) + len(new_variants) >= target_count:
                            break
        
        # Add new variants.
        all_variants.update(new_variants)
        
        if not new_variants:
            # If no new variants are generated, try combined mutations.
            print(f'[train_GP] Iteration {iteration}: single mutations cannot produce more variants; trying combined mutations')
            
            # Mutate existing variants again.
            for expr_str in random_state.sample(list(all_variants), min(50, len(all_variants))):
                if len(all_variants) >= target_count:
                    break
                
                # Combined mutation: change the window first, then the operator.
                window_variants = mutate_time_window(expr_str)
                for wv in window_variants[:3]:  # Limit quantity.
                    op_variants = mutate_operator(wv)
                    for ov in op_variants[:2]:
                        if ov not in all_variants and is_valid_expr(ov):
                            all_variants.add(ov)
                            if len(all_variants) >= target_count:
                                break
        
        print(f'[train_GP] Iteration {iteration}: current variant count={len(all_variants)}')
    
    # Convert to a list and return.
    result = list(all_variants)
    
    # Ensure original seed factors appear at the front of the list.
    for sf in reversed(seed_factors):
        if sf in result:
            result.remove(sf)
            result.insert(0, sf)
    
    print(f'[train_GP] Seed factor augmentation completed: original={len(seed_factors)}, augmented={len(result)}')
    
    return result[:target_count]  # Ensure the target count is not exceeded.


def inject_seed_factors_to_cache(seed_factors: list, data_obj, target_factor_tensor):
    """
    Precompute IC for seed factors and inject them into the cache,
    so GP can use these high-quality factors as building blocks during mutation/crossover.
    
    Args:
        seed_factors: List of seed factor expressions.
        data_obj: Training data object.
        target_factor_tensor: Target factor tensor.
    """
    global cache
    
    injected_count = 0
    for expr_str in seed_factors:
        try:
            expr = eval(expr_str)
            factor = expr.evaluate(data_obj)
            factor = normalize_by_day(factor)
            ic = batch_pearsonr(factor, target_factor_tensor)
            ic = torch.nan_to_num(ic).mean().item()
            
            if not np.isnan(ic) and ic > -1:
                cache[expr_str] = ic
                injected_count += 1
                print(f'[train_GP] Seed factor injected into cache: IC={ic:.4f}, {expr_str[:50]}...')
        except Exception as e:
            print(f'[train_GP] Failed to calculate IC for seed factor: {expr_str[:50]}..., error: {e}')
    
    print(f'[train_GP] Successfully injected {injected_count}/{len(seed_factors)} seed factors into cache')


def convert_alphagen_to_gplearn_program(expr_str, function_set, terminals, n_features):
    """
    Convert an AlphaSAGE expression string into a gplearn program list.
    
    Args:
        expr_str: AlphaSAGE expression string, such as "Add(Ref(close, 2), close)".
        function_set: List of gplearn functions.
        terminals: Terminals list, containing feature names and constant strings.
        n_features: Number of features, excluding constants.
    
    Returns:
        program: gplearn-format program list (a pre-order flattened tree). Nodes should only contain:
          - gplearn _Function objects from function_set.
          - int feature indices in the range [0, n_features).
          - float constants.
        Note: rolling/rolling_binary operator windows in alphagen_generic are already
        "baked into" function names (such as TsMean10 / TsCorr30). Their arity is
        1/2 respectively, so **delta_time should not be appended to the program list**.
    """
    # Parse expression string into an AlphaSAGE expression object.
    expr = eval(expr_str)
    
    # Build a mapping from operator names to gplearn function objects.
    func_name_map = {}
    for func in function_set:
        func_name_map[func.name] = func
    
    # Build a mapping from feature names to terminal indices.
    feature_to_idx = {}
    for i, term in enumerate(terminals):
        if not term.startswith('Constant'):
            # Feature: map to index.
            feature_to_idx[term] = i
            # Handle the special case for open_.
            if term == 'open_':
                feature_to_idx['open'] = i
    
    # Build a mapping from constant values to terminal indices.
    const_to_idx = {}
    const_values = []
    for i, term in enumerate(terminals):
        if term.startswith('Constant'):
            try:
                value = float(term.split('(')[1].split(')')[0])
                const_to_idx[value] = n_features + len(const_values)
                const_values.append(value)
            except:
                pass
    
    # Recursively convert the expression tree into a program list.
    program_list = []
    
    def expr_to_program(expr_obj):
        """Recursively convert an expression object into a program list."""
        from alphagen.data.expression import (
            Feature, Constant, UnaryOperator, BinaryOperator, 
            RollingOperator, PairRollingOperator
        )
        
        # Handle Feature terminal nodes.
        if isinstance(expr_obj, Feature):
            feature_name = expr_obj._feature.name.lower()
            # Map feature name to terminal index.
            if feature_name not in feature_to_idx:
                raise KeyError(
                    f"Unsupported feature '{feature_name}' in seed expression; "
                    f"available terminals: {sorted(feature_to_idx)}"
                )
            idx = feature_to_idx[feature_name]
            program_list.append(idx)
            return
        
        # Handle Constant terminal nodes.
        if isinstance(expr_obj, Constant):
            value = expr_obj._value
            # Look up the constant position in terminals.
            if value in const_to_idx:
                idx = const_to_idx[value]
                program_list.append(idx)
            else:
                # If the constant is not in terminals, use the float value directly.
                program_list.append(float(value))
            return
        
        # Handle operators.
        op_class = type(expr_obj)
        op_name = op_class.__name__
        
        # Handle RollingOperator; the time window is baked into the function name, arity=1.
        if isinstance(expr_obj, RollingOperator):
            delta_time = expr_obj._delta_time
            # Find the function with a matching time window (10, 20, 30, 40, 50).
            # Use the closest time window.
            windows = [10, 20, 30, 40, 50]
            closest_window = min(windows, key=lambda x: abs(x - abs(delta_time)))
            
            func_name = f'{op_name}{closest_window}'
            if func_name in func_name_map:
                func = func_name_map[func_name]
                program_list.append(func)
                # Add operand.
                expr_to_program(expr_obj._operand)
            else:
                # If no matching function is found, use a unary placeholder function (arity=1).
                unary_funcs = [f for f in function_set if f.arity == 1]
                if unary_funcs:
                    program_list.append(unary_funcs[0])  # Use the first unary function as a placeholder.
                    expr_to_program(expr_obj._operand)
                else:
                    # If even a unary function is unavailable, at least add the operand.
                    expr_to_program(expr_obj._operand)
            return
        
        # Handle PairRollingOperator; the time window is baked into the function name, arity=2.
        if isinstance(expr_obj, PairRollingOperator):
            delta_time = expr_obj._delta_time
            windows = [10, 20, 30, 40, 50]
            closest_window = min(windows, key=lambda x: abs(x - abs(delta_time)))
            
            func_name = f'{op_name}{closest_window}'
            if func_name in func_name_map:
                func = func_name_map[func_name]
                program_list.append(func)
                # Add both operands.
                expr_to_program(expr_obj._lhs)
                expr_to_program(expr_obj._rhs)
            else:
                # If no matching function is found, use a binary placeholder function (arity=2).
                binary_funcs = [f for f in function_set if f.arity == 2]
                if binary_funcs:
                    program_list.append(binary_funcs[0])  # Use the first binary function as a placeholder.
                    expr_to_program(expr_obj._lhs)
                    expr_to_program(expr_obj._rhs)
                else:
                    # If even a binary function is unavailable, at least add the operands.
                    expr_to_program(expr_obj._lhs)
                    expr_to_program(expr_obj._rhs)
            return
        
        # Handle UnaryOperator.
        if isinstance(expr_obj, UnaryOperator):
            if op_name in func_name_map:
                func = func_name_map[op_name]
                program_list.append(func)
                expr_to_program(expr_obj._operand)
            else:
                # If no mapping is found, use the first unary function as a placeholder.
                unary_funcs = [f for f in function_set if f.arity == 1]
                if unary_funcs:
                    program_list.append(unary_funcs[0])
                    expr_to_program(expr_obj._operand)
            return
        
        # Handle BinaryOperator.
        if isinstance(expr_obj, BinaryOperator):
            if op_name in func_name_map:
                func = func_name_map[op_name]
                program_list.append(func)
                expr_to_program(expr_obj._lhs)
                expr_to_program(expr_obj._rhs)
            else:
                # If no mapping is found, use the first binary function as a placeholder.
                binary_funcs = [f for f in function_set if f.arity == 2]
                if binary_funcs:
                    program_list.append(binary_funcs[0])
                    expr_to_program(expr_obj._lhs)
                    expr_to_program(expr_obj._rhs)
            return
    
    # Start conversion.
    expr_to_program(expr)
    return program_list


class SeedFactorProgram:
    """
    Wrapper class that wraps a seed factor expression string into a _Program-like object.
    This allows seed factors to be used in the GP initial population and participate in genetic operations.
    
    Note: this class needs to emulate the basic gplearn _Program interface, including the program attribute.
    """
    def __init__(self, expr_str, raw_fitness=None, est_gp=None, terminals=None, n_features=None):
        self.expr_str = expr_str
        self.raw_fitness_ = raw_fitness if raw_fitness is not None else cache.get(expr_str, -1.0)
        self.fitness_ = self.raw_fitness_
        self.parents = {'method': 'Seed Factor', 'parent_idx': None, 'parent_nodes': []}
        self.length_ = expr_str.count('(') + expr_str.count(')')
        self.depth_ = self._estimate_depth()
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None
        self.est_gp = est_gp  # Keep est_gp reference for accessing required attributes.
        self._n_features = n_features  # Keep n_features for later use.
        
        # Convert AlphaSAGE expression into a gplearn program list.
        if est_gp is not None and terminals is not None and n_features is not None:
            try:
                function_set = est_gp._function_set
                self.program = convert_alphagen_to_gplearn_program(
                    expr_str, function_set, terminals, n_features
                )
                
                # Validate whether the program is complete.
                if not self._validate_program_structure(self.program, function_set):
                    print(f'[SeedFactorProgram] Warning: converted program structure is incomplete: {expr_str[:50]}...')
                    # If validation fails, try to create a simple valid program.
                    unary_funcs = [f for f in function_set if f.arity == 1]
                    if unary_funcs:
                        self.program = [unary_funcs[0], 0]  # Simple valid placeholder.
                    else:
                        self.program = [0]  # Single terminal node.
                
                # Update length_ to the actual program length.
                self.length_ = len(self.program)
            except Exception as e:
                print(f'[SeedFactorProgram] Failed to convert program: {expr_str[:50]}..., error: {e}')
                import traceback
                traceback.print_exc()
                # If conversion fails, create a placeholder program.
                unary_funcs = [f for f in est_gp._function_set if f.arity == 1] if est_gp._function_set else []
                if unary_funcs:
                    self.program = [unary_funcs[0], 0]  # Simple valid placeholder.
                else:
                    self.program = [0]
        else:
            # Delay initialization and set it later.
            self.program = None
            self._terminals = terminals
            self._n_features = n_features
    
    def _estimate_depth(self):
        """Estimate expression depth."""
        depth = 0
        max_depth = 0
        for char in self.expr_str:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
        return max_depth
    
    def _validate_program_structure(self, program, function_set):
        """Validate whether the program structure is complete, similar to gplearn validate_program."""
        if not program or len(program) == 0:
            return False
        
        # Check function nodes by testing whether they have an arity attribute.
        terminals = [0]
        for node in program:
            # Check whether this is a function node.
            if hasattr(node, 'arity'):
                arity = node.arity
                terminals.append(arity)
            else:
                # Terminal node (int or float).
                if len(terminals) == 0:
                    return False
                terminals[-1] -= 1
                while len(terminals) > 0 and terminals[-1] == 0:
                    terminals.pop()
                    if len(terminals) > 0:
                        terminals[-1] -= 1
        
        # Validation result: terminals should equal [-1], meaning all functions have enough arguments.
        return terminals == [-1]
    
    def __str__(self):
        """Return the expression string so the _metric function can identify it."""
        return self.expr_str
    
    def execute(self, X):
        """Execute the expression and return factor values."""
        try:
            expr = eval(self.expr_str)
            factor = expr.evaluate(data)
            factor = normalize_by_day(factor)
            # Convert to numpy array format for gplearn compatibility.
            if hasattr(factor, 'cpu'):
                factor_np = factor.cpu().numpy()
            else:
                factor_np = np.array(factor)
            # Ensure a one-dimensional array is returned.
            if len(factor_np.shape) > 1:
                factor_np = factor_np.flatten()
            return factor_np
        except Exception as e:
            print(f'[SeedFactorProgram] Execution failed: {self.expr_str[:50]}..., error: {e}')
            return np.zeros(X.shape[0]) if len(X.shape) > 0 else np.array([0.0])
    
    def fitness(self, parsimony_coefficient=None):
        """Calculate fitness."""
        if parsimony_coefficient is None:
            parsimony_coefficient = 0.0  # Do not use parsimony by default.
        penalty = parsimony_coefficient * self.length_ * (1 if self.raw_fitness_ >= 0 else -1)
        return self.raw_fitness_ - penalty
    
    def get_all_indices(self, n_samples=None, max_samples=None, random_state=None):
        """Emulate the _Program get_all_indices method."""
        # Return indices for all samples without subsampling.
        if n_samples is None:
            return np.arange(self._n_samples) if self._n_samples else np.array([]), np.array([])
        indices = np.arange(n_samples)
        return indices, np.array([])
    
    def reproduce(self):
        """Copy itself for the reproduction operation.

        gplearn expects genetic operators to return a flattened program list,
        not a Program object. _Program.reproduce() also returns copy(self.program).
        """
        from copy import copy
        return copy(self.program) if self.program is not None else []
    
    def get_subtree(self, random_state, program=None):
        """Get a random subtree, consistent with gplearn's native implementation.
        
        Algorithm:
        - In the pre-order flattened representation, a subtree is contiguous.
        - stack means "how many more nodes are needed to complete the current subtree".
        - Initial stack=1, meaning one complete subtree is required.
        - When a function node is encountered, stack += arity.
        - end += 1 after each node is processed.
        - When stack == end - start, the subtree is exactly complete.
        """
        if program is None:
            program = self.program
        
        if program is None or len(program) == 0:
            return 0, 1
        
        # Select a function node with 90% probability or a terminal node with 10% probability.
        # Follow the Koza (1992) method.
        func_indices = [i for i, node in enumerate(program) 
                       if hasattr(node, 'arity')]
        
        if len(func_indices) == 0:
            # If there are no function nodes, only terminals, return the whole program.
            return 0, len(program)
        
        # 90% probability to select a function node, 10% to select a terminal node.
        if random_state.uniform() < 0.9:
            start = func_indices[random_state.randint(len(func_indices))]
        else:
            # Select a random position.
            start = random_state.randint(len(program))
        
        # Find the subtree range, consistent with gplearn's native implementation.
        stack = 1
        end = start
        while stack > end - start:
            if end >= len(program):
                # Prevent out-of-bounds access by returning up to the program end.
                break
            node = program[end]
            # Only function nodes increase stack; terminal nodes do not change stack.
            if hasattr(node, 'arity'):
                stack += node.arity
            end += 1
        
        return start, end
    
    def crossover(self, donor, random_state):
        """Execute crossover."""
        from copy import copy
        
        if self.program is None or len(self.program) == 0:
            # If program is empty, return itself.
            return copy(self.program) if self.program else [], [], []
        
        # Get subtree.
        start, end = self.get_subtree(random_state)
        # Ensure end > start.
        if end <= start:
            end = min(start + 1, len(self.program))
        removed = list(range(start, end))
        
        # Handle donor, which may be a _Program object, SeedFactorProgram, or list.
        if hasattr(donor, 'program'):
            donor_program = donor.program
        elif isinstance(donor, list):
            donor_program = donor
        else:
            # If it cannot be handled, return itself.
            return copy(self.program), removed, []
        
        if donor_program is None or len(donor_program) == 0:
            return copy(self.program), removed, []
        
        # Get donor subtree.
        donor_start, donor_end = self.get_subtree(random_state, donor_program)
        # Ensure donor_end > donor_start.
        if donor_end <= donor_start:
            donor_end = min(donor_start + 1, len(donor_program))
        donor_removed = list(set(range(len(donor_program))) - set(range(donor_start, donor_end)))
        
        # Execute crossover.
        new_program = copy(self.program[:start]) + copy(donor_program[donor_start:donor_end]) + copy(self.program[end:])
        
        # Validate the new program structure.
        if not self._validate_program_structure(new_program, self.est_gp._function_set if self.est_gp else []):
            # If the crossed-over program is incomplete, return itself (reproduction).
            return copy(self.program), [], []
        
        return new_program, removed, donor_removed
    
    def subtree_mutation(self, random_state):
        """Execute subtree mutation.
        
        Generate a random subtree (chicken) to replace a random subtree in the current program.
        """
        # Generate a random subtree as the donor.
        chicken = self._generate_random_subtree(random_state)
        
        # Use crossover to implement subtree mutation.
        return self.crossover(chicken, random_state)
    
    def _generate_random_subtree(self, random_state, max_depth=4):
        """Generate a random subtree.
        
        Args:
            random_state: Random number generator.
            max_depth: Maximum subtree depth.
            
        Returns:
            list: Program list for the random subtree.
        """
        if self.est_gp is None or not self.est_gp._function_set:
            # If there is no function set, return a simple terminal.
            n_features = self._n_features if self._n_features else 5
            return [random_state.randint(n_features)]
        
        n_features = self._n_features
        if n_features is None:
            if hasattr(self.est_gp, 'n_features_in_'):
                n_features = self.est_gp.n_features_in_
            elif hasattr(self.est_gp, 'n_features'):
                n_features = self.est_gp.n_features
            else:
                n_features = 5
        
        function_set = self.est_gp._function_set
        
        # Try to generate a random program using gplearn's _Program class.
        try:
            from gplearn._program import _Program
            temp_program = _Program(
                function_set=function_set,
                arities=self.est_gp._arities,
                init_depth=(2, max_depth),
                init_method='half and half',
                n_features=n_features,
                const_range=self.est_gp.const_range,
                metric=self.est_gp._metric,
                p_point_replace=self.est_gp.p_point_replace,
                parsimony_coefficient=self.est_gp.parsimony_coefficient,
                random_state=random_state
            )
            return temp_program.program
        except Exception:
            pass
        
        # If _Program cannot be used, manually generate a random subtree.
        return self._build_random_tree(random_state, function_set, n_features, max_depth)
    
    def _build_random_tree(self, random_state, function_set, n_features, max_depth, current_depth=0):
        """Recursively build a random tree.
        
        Args:
            random_state: Random number generator.
            function_set: Function set.
            n_features: Number of features.
            max_depth: Maximum depth.
            current_depth: Current depth.
            
        Returns:
            list: Program list for the random tree in pre-order traversal.
        """
        program = []
        
        # Generate a terminal if max depth is reached or randomness selects a terminal.
        if current_depth >= max_depth or (current_depth > 0 and random_state.uniform() < 0.3):
            # Generate terminal node (feature index).
            terminal = random_state.randint(n_features)
            return [terminal]
        
        # Generate function node.
        func = function_set[random_state.randint(len(function_set))]
        program.append(func)
        
        # Recursively generate child nodes.
        for _ in range(func.arity):
            child = self._build_random_tree(
                random_state, function_set, n_features, max_depth, current_depth + 1
            )
            program.extend(child)
        
        return program
    
    def hoist_mutation(self, random_state):
        """Execute hoist mutation."""
        from copy import copy
        
        if self.program is None or len(self.program) == 0:
            return copy(self.program) if self.program else [], []
        
        # Get subtree.
        start, end = self.get_subtree(random_state)
        if end <= start or end > len(self.program):
            # If subtree is invalid, return itself.
            return copy(self.program), []
        
        subtree = copy(self.program[start:end])
        
        # If the subtree is too short, hoist mutation cannot be performed.
        if len(subtree) <= 1:
            return copy(self.program), []
        
        # Get a subtree of the subtree.
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        if sub_end <= sub_start or sub_end > len(subtree):
            return copy(self.program), []
        
        hoist = copy(subtree[sub_start:sub_end])
        
        # Determine removed nodes.
        removed = list(set(range(start, end)) - set(range(start + sub_start, start + sub_end)))
        
        # Execute hoist.
        new_program = copy(self.program[:start]) + hoist + copy(self.program[end:])
        
        # Validate the new program structure.
        if not self._validate_program_structure(new_program, self.est_gp._function_set if self.est_gp else []):
            # If the hoisted program is incomplete, return itself (reproduction).
            return copy(self.program), []
        
        return new_program, removed
    
    def point_mutation(self, random_state):
        """Execute point mutation."""
        from copy import copy
        
        new_program = copy(self.program) if self.program else []
        mutated = []
        
        if not self.est_gp:
            return new_program, mutated
        
        # Randomly select nodes to mutate.
        for i in range(len(new_program)):
            if random_state.uniform() < (self.est_gp.p_point_replace if hasattr(self.est_gp, 'p_point_replace') else 0.1):
                node = new_program[i]
                
                # Check whether this is a function node by testing for the arity attribute.
                if hasattr(node, 'arity'):
                    # Replace with another function of the same arity.
                    same_arity_funcs = [f for f in self.est_gp._function_set if f.arity == node.arity]
                    if same_arity_funcs:
                        new_program[i] = same_arity_funcs[random_state.randint(len(same_arity_funcs))]
                        mutated.append(i)
                elif isinstance(node, int):
                    # Replace with another feature or constant.
                    # Get n_features, preferring the saved value.
                    n_features = self._n_features
                    if n_features is None:
                        if hasattr(self.est_gp, 'n_features_in_'):
                            n_features = self.est_gp.n_features_in_
                        elif hasattr(self.est_gp, 'n_features'):
                            n_features = self.est_gp.n_features
                        else:
                            n_features = 5  # Default value.
                    
                    if self.est_gp.const_range is not None:
                        # Can be either a feature or a constant.
                        if random_state.uniform() < 0.5:
                            new_program[i] = random_state.randint(n_features)
                        else:
                            new_program[i] = random_state.uniform(*self.est_gp.const_range)
                    else:
                        # Can only be a feature.
                        new_program[i] = random_state.randint(n_features)
                    mutated.append(i)
                elif isinstance(node, float):
                    # Replace with another constant.
                    if self.est_gp.const_range is not None:
                        new_program[i] = random_state.uniform(*self.est_gp.const_range)
                        mutated.append(i)
        
        # Validate the new program structure.
        if not self._validate_program_structure(new_program, self.est_gp._function_set if self.est_gp else []):
            # If the mutated program is incomplete, return itself (reproduction).
            return copy(self.program) if self.program else [], []
        
        return new_program, mutated


def inject_seed_factors_to_initial_population(est_gp, seed_factors, X_train, y_train, terminals, n_features, replace_ratio=0.1):
    """
    Inject seed factors into the GP initial population by replacing some random individuals.
    
    Args:
        est_gp: SymbolicRegressor instance that has already been fit once.
        seed_factors: List of seed factor expression strings.
        X_train: Training data X.
        y_train: Training data y.
        replace_ratio: Ratio of the initial population to replace (between 0 and 1).
    """
    if not seed_factors or not hasattr(est_gp, '_programs') or len(est_gp._programs) == 0:
        print(f'[train_GP] Cannot inject seed factors: seed factors are empty or GP is not initialized')
        return 0
    
    initial_population = est_gp._programs[0]
    if initial_population is None or len(initial_population) == 0:
        print(f'[train_GP] Cannot inject seed factors: initial population is empty')
        return 0
    
    # Calculate the number of individuals to replace.
    n_replace_by_ratio = int(len(initial_population) * replace_ratio)
    n_replace = min(len(seed_factors), n_replace_by_ratio, len(initial_population))
    if n_replace == 0:
        n_replace = min(len(seed_factors), len(initial_population))
    
    print(f'[train_GP] Preparing to inject {n_replace} seed factors into the initial population (total: {len(initial_population)}, replacement ratio: {replace_ratio:.1%})')
    
    # Randomly select individual indices to replace, preferring lower-fitness individuals.
    if hasattr(est_gp, 'random_state') and est_gp.random_state is not None:
        random_state = est_gp.random_state
    else:
        random_state = np.random.RandomState(42)
    
    # Sort by fitness and prioritize replacing lower-fitness individuals.
    fitness_scores = [p.raw_fitness_ if hasattr(p, 'raw_fitness_') and p.raw_fitness_ is not None else -999 
                      for p in initial_population]
    sorted_indices = np.argsort(fitness_scores)  # Ascending order, lower fitness first.
    
    # Select the n_replace individuals with the lowest fitness for replacement.
    replace_indices = sorted_indices[:n_replace]
    
    # Create seed factor program objects and replace individuals.
    replaced_count = 0
    for i, idx in enumerate(replace_indices):
        if i >= len(seed_factors):
            break
        
        expr_str = seed_factors[i]
        # Get the seed factor's IC value; it should already be in cache because
        # inject_seed_factors_to_cache has calculated it.
        seed_ic = cache.get(expr_str, None)
        
        if seed_ic is None:
            # If it is not in cache, try to calculate it. This should not happen because
            # inject_seed_factors_to_cache should have already handled it.
            print(f'[train_GP] Warning: seed factor {i} is not in cache; trying to calculate it...')
            try:
                expr = eval(expr_str)
                factor = expr.evaluate(data)
                factor = normalize_by_day(factor)
                seed_ic = batch_pearsonr(factor, target_factor)
                seed_ic = torch.nan_to_num(seed_ic).mean().item()
                cache[expr_str] = seed_ic
            except Exception as e:
                print(f'[train_GP] Failed to calculate IC for seed factor {i}: {e}; skipping this factor')
                continue  # IC calculation failed; skip this factor and do not inject it.
        
        # Skip factors with invalid IC values, such as failures previously set to -1.0.
        if seed_ic <= -1.0 or np.isnan(seed_ic):
            print(f'[train_GP] Seed factor {i} has invalid IC ({seed_ic}); skipping this factor')
            continue
        
        # Create seed factor program object with terminals and n_features.
        seed_program = SeedFactorProgram(
            expr_str, raw_fitness=seed_ic, est_gp=est_gp,
            terminals=terminals, n_features=n_features
        )
        
        # Set required attributes to emulate a _Program object.
        if hasattr(initial_population[0], '_n_samples'):
            seed_program._n_samples = initial_population[0]._n_samples
        if hasattr(initial_population[0], '_max_samples'):
            seed_program._max_samples = initial_population[0]._max_samples
        
        # Replace the individual in the initial population.
        old_fitness = fitness_scores[idx]
        initial_population[idx] = seed_program
        replaced_count += 1
        print(f'[train_GP] Replaced initial population individual {idx}: old_fitness={old_fitness:.4f} -> new_fitness={seed_ic:.4f}, {expr_str[:60]}...')
    
    print(f'[train_GP] Successfully injected {replaced_count} seed factors into the initial population')
    return replaced_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, default='csi300')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-end-year', type=int, default=2020)
    parser.add_argument('--freq', type=str, default='day')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--target-horizon-days', type=int, default=10,
                        help='Trading-day length of the holding-period return target; 10 means Ref(close, -11)/Ref(close, -1)-1')
    parser.add_argument('--seed-factors-file', type=str, default='',
                        help='Path to the seed factor JSON file used to guide the GP initial population')
    args = parser.parse_args()
    run(args)
