"""FastAPI-based MCP IC Calculation Service

MCP (Model Context Protocol) service providing factor IC coefficient calculation functionality.
Accepts factor expressions, datasets, and time ranges, returns IC-related metrics.

Usage:
    # Start service
    uvicorn mcp_ic_service:app --host 0.0.0.0 --port 8000

    # Or run directly
    python mcp_ic_service.py

API Endpoints:
    POST /calculate_ic - Calculate factor IC metrics
    GET /tools - Get available tool list (MCP protocol)
    POST /tools/call - Call tool (MCP protocol)
"""

import os
import re
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Data Models
# ============================================================================

class ICRequest(BaseModel):
    """IC calculation request parameters"""
    formula: str = Field(..., description="Qlib format factor expression, e.g., 'Ts_Mean($close, 10) / Ts_Mean($close, 30) - 1'")
    instruments: str = Field(default="csi300", description="Stock pool name: csi300, csi500, csi800, csi1000, etc.")
    start_date: str = Field(default="2020-01-01", description="Start date (YYYY-MM-DD)")
    end_date: str = Field(default="2023-12-31", description="End date (YYYY-MM-DD)")
    label_expr: str = Field(
        default="Ref($close, -11)/Ref($close, -1) - 1",
        description="Label expression for calculating future returns"
    )
    provider_uri: Optional[str] = Field(default="~/.qlib/qlib_data/cn_data", description="Qlib data path, default auto-detect (qlib_bin directory)")


class ICMetrics(BaseModel):
    """IC metrics results"""
    ic_mean: float = Field(..., description="IC mean - Mean Pearson correlation coefficient between factor and future returns")
    ic_std: float = Field(..., description="IC standard deviation - Standard deviation of IC series, measures stability")
    ir: float = Field(..., description="Information Ratio (IR) - IC mean / IC std, risk-adjusted return")
    rank_ic_mean: float = Field(..., description="Rank IC mean - IC mean calculated using Spearman rank correlation")
    rank_ic_std: float = Field(..., description="Rank IC standard deviation")
    rank_ir: float = Field(..., description="Rank IR - Rank IC mean / Rank IC std")
    ic_win_rate: float = Field(..., description="IC win rate - Percentage of trading days with IC > 0")


class ICResponse(BaseModel):
    """IC calculation response"""
    success: bool = Field(..., description="Whether calculation succeeded")
    metrics: Optional[ICMetrics] = Field(None, description="IC metrics results")
    message: str = Field(default="", description="Status message")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed information")


class MCPToolParameter(BaseModel):
    """MCP tool parameter definition"""
    name: str
    description: str
    type: str
    required: bool = True
    default: Optional[Any] = None


class MCPTool(BaseModel):
    """MCP tool definition"""
    name: str
    description: str
    parameters: List[MCPToolParameter]


class MCPToolsResponse(BaseModel):
    """MCP tool list response"""
    tools: List[MCPTool]


class MCPToolCallRequest(BaseModel):
    """MCP tool call request"""
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default={}, description="Tool parameters")


class MCPToolCallResponse(BaseModel):
    """MCP tool call response"""
    content: List[Dict[str, Any]]
    isError: bool = False


# ============================================================================
# IC Calculation Core Logic (self-contained, no external local file references)
# ============================================================================


def preprocess_formula(formula: str) -> str:
    """Preprocess factor expression, convert to Qlib-supported format
    
    Main processing:
    1. Replace Constant(x) with direct number x
    2. Handle other possible format differences
    
    Args:
        formula: Original factor expression
        
    Returns:
        Converted Qlib-compatible expression
    """
    if not formula:
        return formula
    
    original_formula = formula
    
    # 1. Replace Constant(number) with direct number
    # Match Constant(1.0), Constant(-2.5), Constant(100), etc.
    constant_pattern = r'Constant\s*\(\s*(-?\d+\.?\d*)\s*\)'
    formula = re.sub(constant_pattern, r'\1', formula)
    
    # 2. Handle possible extra spaces
    formula = re.sub(r'\s+', ' ', formula).strip()
    
    # 3. Handle some common operator aliases (if needed)
    # Qlib uses Add, Sub, Mul, Div, etc., but also supports +, -, *, /
    
    if formula != original_formula:
        logger.info(f"Expression preprocessing: {original_formula} -> {formula}")
    
    return formula


def compute_ic_series(
    factor_values: pd.Series,
    labels: pd.Series,
    method: str = 'pearson'
) -> pd.Series:
    """Calculate IC series for each cross-section (Panel format data)
    
    Args:
        factor_values: Factor values (MultiIndex: datetime, instrument)
        labels: Return labels (MultiIndex: datetime, instrument)
        method: 'pearson' or 'spearman'
        
    Returns:
        IC series (by date)
    """
    df = pd.concat([factor_values, labels], axis=1, keys=['factor', 'label']).dropna()
    
    def calc_ic(group):
        if len(group) < 3:
            return np.nan
        try:
            if method == 'pearson':
                ic, _ = pearsonr(group['factor'], group['label'])
            else:
                ic, _ = spearmanr(group['factor'], group['label'])
            return ic
        except Exception:
            return np.nan
    
    return df.groupby(level=0).apply(calc_ic)


def compute_factor_metrics(
    factor_values: pd.Series,
    labels: pd.Series
) -> Dict[str, Any]:
    """Calculate factor evaluation metrics
    
    Args:
        factor_values: Factor values
        labels: Return labels
        
    Returns:
        Metrics dictionary, containing:
        - ic_mean: IC mean
        - ic_std: IC standard deviation
        - ir: Information ratio
        - rank_ic_mean: Rank IC mean
        - rank_ic_std: Rank IC standard deviation
        - rank_ir: Rank IR
        - ic_win_rate: IC win rate
    """
    ic_series = compute_ic_series(factor_values, labels, method='pearson')
    rank_ic_series = compute_ic_series(factor_values, labels, method='spearman')
    
    # Filter out NaN values
    ic_series_clean = ic_series.dropna()
    rank_ic_series_clean = rank_ic_series.dropna()
    
    # Calculate IC mean and standard deviation
    ic_mean = float(ic_series_clean.mean()) if len(ic_series_clean) > 0 else 0.0
    ic_std = float(ic_series_clean.std()) if len(ic_series_clean) > 0 else 0.0
    
    # Calculate Rank IC mean and standard deviation
    rank_ic_mean = float(rank_ic_series_clean.mean()) if len(rank_ic_series_clean) > 0 else 0.0
    rank_ic_std = float(rank_ic_series_clean.std()) if len(rank_ic_series_clean) > 0 else 0.0
    
    # Information ratio
    ir = ic_mean / ic_std if ic_std > 1e-8 else 0.0
    rank_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-8 else 0.0
    
    # IC win rate
    ic_win_rate = float((ic_series_clean > 0).mean()) if len(ic_series_clean) > 0 else 0.0
    
    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ir': ir,
        'rank_ic_mean': rank_ic_mean,
        'rank_ic_std': rank_ic_std,
        'rank_ir': rank_ir,
        'ic_win_rate': ic_win_rate,
        'ic_series': ic_series_clean,
        'rank_ic_series': rank_ic_series_clean,
        'sample_count': len(ic_series_clean),
    }


def init_qlib(provider_uri: Optional[str] = "~/.qlib/qlib_data/cn_data") -> tuple:
    """Initialize Qlib
    
    Args:
        provider_uri: Qlib data path
        
    Returns:
        (Whether initialization succeeded, actual data path used)
    """
    try:
        import qlib
        from qlib.config import REG_CN
        
        # Expand ~ path
        if provider_uri and provider_uri.startswith('~'):
            provider_uri = os.path.expanduser(provider_uri)
        
        # Auto-detect data path
        if provider_uri is None or not os.path.exists(provider_uri):
            possible_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qlib_bin'),
                'qlib_bin',
                os.path.abspath('qlib_bin'),
                r'C:\edge_download\python_project\Qlib_project\qlib_bin',
                r'C:\edge_download\python_project\Qlib_project\MCTS_QCM\qlib_bin',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    provider_uri = path
                    logger.info(f"Auto-detected data path: {provider_uri}")
                    break
        
        if provider_uri is None or not os.path.exists(provider_uri):
            logger.error("Qlib data path not found")
            return False, None
        
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        logger.info(f"Qlib initialized successfully, data path: {provider_uri}")
        return True, provider_uri
        
    except Exception as e:
        logger.error(f"Qlib initialization failed: {e}")
        return False, None


def get_instruments_list(
    instruments: str,
    start_date: str,
    end_date: str,
    provider_uri: Optional[str] = None
) -> List[str]:
    """Get stock list
    
    Args:
        instruments: Stock pool name
        start_date: Start date
        end_date: End date
        provider_uri: Data path
        
    Returns:
        Stock code list
    """
    instrument_list = None
    
    # Method 1: Use D.list_instruments() to get stock list
    try:
        from qlib.data import D
        
        # Use list_instruments method to get stock list
        instrument_list = D.list_instruments(
            instruments=D.instruments(instruments),
            start_time=start_date,
            end_time=end_date,
            as_list=True
        )
        
        if instrument_list and len(instrument_list) > 0:
            logger.info(f"Got {len(instrument_list)} stocks using D.list_instruments")
            return list(instrument_list)
            
    except Exception as e:
        logger.warning(f"D.list_instruments failed: {e}, trying to read from file")
    
    # Method 2: Read directly from file
    logger.info("Trying to read stock list directly from file...")
    
    # Determine provider_uri
    if provider_uri is None:
        # Use globally saved path
        global _qlib_provider_uri
        provider_uri = _qlib_provider_uri
    
    if provider_uri is None:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qlib_bin'),
            'qlib_bin',
            os.path.abspath('qlib_bin'),
            r'C:\edge_download\python_project\Qlib_project\qlib_bin',
            r'C:\edge_download\python_project\Qlib_project\MCTS_QCM\qlib_bin',
        ]
        for path in possible_paths:
            test_file = os.path.join(path, 'instruments', f'{instruments}.txt')
            if os.path.exists(test_file):
                provider_uri = path
                logger.info(f"Found data path: {provider_uri}")
                break
    
    if provider_uri:
        file_path = os.path.join(provider_uri, 'instruments', f'{instruments}.txt')
        logger.info(f"Trying to read stock pool file: {file_path}")
        
        if os.path.exists(file_path):
            stocks_from_file = set()
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 1:
                            stock_code = parts[0]
                            if len(parts) >= 3:
                                stock_start = parts[1]
                                stock_end = parts[2]
                                # Check if date ranges intersect
                                if stock_start <= end_date and stock_end >= start_date:
                                    stocks_from_file.add(stock_code)
                            else:
                                stocks_from_file.add(stock_code)
            
            if len(stocks_from_file) > 0:
                instrument_list = sorted(list(stocks_from_file))
                logger.info(f"Read {len(instrument_list)} stocks from file")
        else:
            logger.error(f"Stock pool file does not exist: {file_path}")
    else:
        logger.error("Data path not found, cannot read stock pool file")
    
    return instrument_list or []


def calculate_ic(
    formula: str,
    instruments: str,
    start_date: str,
    end_date: str,
    label_expr: str = 'Ref($close, -11)/Ref($close, -1) - 1',
    provider_uri: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate factor IC metrics
    
    Args:
        formula: Qlib format factor expression
        instruments: Stock pool name
        start_date: Start date
        end_date: End date
        label_expr: Label expression
        provider_uri: Qlib data path
        
    Returns:
        IC metrics dictionary
    """
    try:
        from qlib.data import D
        
        # Save original expressions
        original_formula = formula
        original_label_expr = label_expr
        
        # Preprocess expressions, convert Constant(x) to Qlib-supported format
        formula = preprocess_formula(formula)
        label_expr = preprocess_formula(label_expr)
        
        logger.info(f"Calculating factor: {formula}")
        logger.info(f"Stock pool: {instruments}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Label expression: {label_expr}")
        
        # Use globally saved provider_uri
        global _qlib_provider_uri
        if provider_uri is None:
            provider_uri = _qlib_provider_uri
        
        # Get stock list
        instrument_list = get_instruments_list(instruments, start_date, end_date, provider_uri)
        
        if len(instrument_list) == 0:
            raise ValueError(f"Stock pool {instruments} did not retrieve any stocks")
        
        logger.info(f"Retrieved {len(instrument_list)} stocks")
        
        # Calculate factor values
        logger.info("Calculating factor values...")
        factor_df = D.features(
            instrument_list,
            [formula],
            start_time=start_date,
            end_time=end_date
        )
        
        # Calculate label values
        logger.info("Calculating label values...")
        label_df = D.features(
            instrument_list,
            [label_expr],
            start_time=start_date,
            end_time=end_date
        )
        
        # Convert to Series format
        if isinstance(factor_df, pd.DataFrame):
            factor_values = factor_df.iloc[:, 0]
        else:
            factor_values = factor_df
        
        if isinstance(label_df, pd.DataFrame):
            labels = label_df.iloc[:, 0]
        else:
            labels = label_df
        
        # Ensure index names are consistent
        if isinstance(factor_values.index, pd.MultiIndex):
            factor_values.index.names = ['datetime', 'instrument']
        if isinstance(labels.index, pd.MultiIndex):
            labels.index.names = ['datetime', 'instrument']
        
        logger.info(f"Factor value calculation completed, {len(factor_values)} records")
        logger.info(f"Label value calculation completed, {len(labels)} records")
        
        # Align factor values and labels
        aligned = pd.concat([factor_values, labels], axis=1, keys=['factor', 'label']).dropna()
        
        if len(aligned) < 10:
            raise ValueError(f"Aligned data too sparse: {len(aligned)}, please check if expressions are correct")
        
        logger.info(f"Aligned data count: {len(aligned)} records")
        
        # Extract aligned data
        factor_aligned = aligned['factor']
        labels_aligned = aligned['label']
        
        # Calculate IC metrics
        logger.info("Calculating IC metrics...")
        metrics = compute_factor_metrics(factor_aligned, labels_aligned)
        
        return {
            'success': True,
            'metrics': {
                'ic_mean': float(metrics['ic_mean']),
                'ic_std': float(metrics['ic_std']),
                'ir': float(metrics['ir']),
                'rank_ic_mean': float(metrics['rank_ic_mean']),
                'rank_ic_std': float(metrics['rank_ic_std']),
                'rank_ir': float(metrics['rank_ir']),
                'ic_win_rate': float(metrics['ic_win_rate']),
            },
            'details': {
                'formula': formula,
                'original_formula': original_formula,
                'instruments': instruments,
                'start_date': start_date,
                'end_date': end_date,
                'label_expr': label_expr,
                'original_label_expr': original_label_expr,
                'stock_count': len(instrument_list),
                'sample_count': metrics.get('sample_count', 0),
                'data_records': len(aligned),
            }
        }
        
    except ImportError:
        logger.error("Qlib not installed, please install: pip install pyqlib")
        raise HTTPException(status_code=500, detail="Qlib not installed")
    except Exception as e:
        logger.error(f"Error calculating IC: {e}", exc_info=True)
        raise


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="MCP IC Calculation Service",
    description="""
MCP (Model Context Protocol) based factor IC calculation service.

Features:
- Calculate IC (Information Coefficient) between factors and returns
- Support Qlib format factor expressions
- Support multiple stock pools (csi300, csi500, csi800, csi1000, etc.)
- Return complete IC metrics: IC mean, IC std, IR, Rank IC, IC win rate, etc.

MCP Protocol Endpoints:
- GET /tools: Get available tool list
- POST /tools/call: Call tool
    """,
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qlib initialization status
_qlib_initialized = False
_qlib_provider_uri: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Qlib when application starts"""
    global _qlib_initialized, _qlib_provider_uri
    _qlib_initialized, _qlib_provider_uri = init_qlib()
    if _qlib_initialized:
        logger.info(f"Service started successfully, Qlib initialized, data path: {_qlib_provider_uri}")
    else:
        logger.warning("Service started, but Qlib initialization failed")


# ============================================================================
# MCP Protocol Endpoints
# ============================================================================

@app.get("/tools", response_model=MCPToolsResponse)
async def get_tools():
    """Get available tool list (MCP protocol)"""
    tools = [
        MCPTool(
            name="calculate_ic",
            description="Calculate factor IC (Information Coefficient) metrics. Returns IC mean, IC std, Information Ratio (IR), Rank IC mean, Rank IC std, Rank IR, IC win rate.",
            parameters=[
                MCPToolParameter(
                    name="formula",
                    description="Qlib format factor expression, e.g., 'Ts_Mean($close, 10) / Ts_Mean($close, 30) - 1'",
                    type="string",
                    required=True
                ),
                MCPToolParameter(
                    name="instruments",
                    description="Stock pool name: csi300, csi500, csi800, csi1000, all, etc.",
                    type="string",
                    required=False,
                    default="csi300"
                ),
                MCPToolParameter(
                    name="start_date",
                    description="Start date (YYYY-MM-DD)",
                    type="string",
                    required=False,
                    default="2020-01-01"
                ),
                MCPToolParameter(
                    name="end_date",
                    description="End date (YYYY-MM-DD)",
                    type="string",
                    required=False,
                    default="2023-12-31"
                ),
                MCPToolParameter(
                    name="label_expr",
                    description="Label expression for calculating future returns",
                    type="string",
                    required=False,
                    default="Ref($close, -11)/Ref($close, -1) - 1"
                ),
            ]
        )
    ]
    return MCPToolsResponse(tools=tools)


@app.post("/tools/call", response_model=MCPToolCallResponse)
async def call_tool(request: MCPToolCallRequest):
    """Call tool (MCP protocol)"""
    global _qlib_initialized, _qlib_provider_uri
    
    if request.name == "calculate_ic":
        try:
            # Ensure Qlib is initialized
            if not _qlib_initialized:
                provider_uri = request.arguments.get("provider_uri")
                _qlib_initialized, _qlib_provider_uri = init_qlib(provider_uri)
                if not _qlib_initialized:
                    return MCPToolCallResponse(
                        content=[{
                            "type": "text",
                            "text": "Error: Qlib initialization failed, please check data path configuration"
                        }],
                        isError=True
                    )
            
            # Extract parameters
            formula = request.arguments.get("formula")
            if not formula:
                return MCPToolCallResponse(
                    content=[{
                        "type": "text",
                        "text": "Error: Missing required parameter 'formula'"
                    }],
                    isError=True
                )
            
            instruments = request.arguments.get("instruments", "csi300")
            start_date = request.arguments.get("start_date", "2020-01-01")
            end_date = request.arguments.get("end_date", "2023-12-31")
            label_expr = request.arguments.get("label_expr", "Ref($close, -11)/Ref($close, -1) - 1")
            provider_uri = request.arguments.get("provider_uri")
            
            # Calculate IC
            result = calculate_ic(
                formula=formula,
                instruments=instruments,
                start_date=start_date,
                end_date=end_date,
                label_expr=label_expr,
                provider_uri=provider_uri
            )
            
            # Format output
            metrics = result['metrics']
            details = result['details']
            
            output_text = f"""IC Calculation Results
================================================================================
Factor Expression: {details['formula']}
Stock Pool: {details['instruments']} ({details['stock_count']} stocks)
Date Range: {details['start_date']} to {details['end_date']}
Data Records: {details['data_records']}
================================================================================

[IC Metrics]
  IC Mean (IC_mean):        {metrics['ic_mean']:.6f}
  IC Std (IC_std):          {metrics['ic_std']:.6f}
  Information Ratio (IR):   {metrics['ir']:.6f}
  Rank IC Mean:             {metrics['rank_ic_mean']:.6f}
  Rank IC Std:              {metrics['rank_ic_std']:.6f}
  Rank IR:                  {metrics['rank_ir']:.6f}
  IC Win Rate:              {metrics['ic_win_rate']:.2%}

================================================================================
"""
            
            return MCPToolCallResponse(
                content=[
                    {
                        "type": "text",
                        "text": output_text
                    },
                    {
                        "type": "json",
                        "data": result
                    }
                ],
                isError=False
            )
            
        except Exception as e:
            logger.error(f"Error calculating IC: {e}", exc_info=True)
            return MCPToolCallResponse(
                content=[{
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }],
                isError=True
            )
    else:
        return MCPToolCallResponse(
            content=[{
                "type": "text",
                "text": f"Unknown tool: {request.name}"
            }],
            isError=True
        )


# ============================================================================
# Module Interface Function (for calling by other modules)
# ============================================================================

def run(args) -> None:
    """
    Module interface function for calling by other modules
    
    Args:
        args: SimpleNamespace object containing the following attributes:
            - formula: Qlib format factor expression (required)
            - instruments: Stock pool name, default "csi300"
            - start_date: Start date, default "2020-01-01"
            - end_date: End date, default "2023-12-31"
            - label_expr: Label expression, default "Ref($close, -11)/Ref($close, -1) - 1"
            - provider_uri: Qlib data path, default None (auto-detect)
    """
    try:
        # Get parameters from args
        formula = getattr(args, "formula", None)
        if not formula:
            raise ValueError("formula parameter is required")
        
        instruments = getattr(args, "instruments", "csi300")
        start_date = getattr(args, "start_date", "2020-01-01")
        end_date = getattr(args, "end_date", "2023-12-31")
        label_expr = getattr(args, "label_expr", "Ref($close, -11)/Ref($close, -1) - 1")
        provider_uri = getattr(args, "provider_uri", None)
        
        # Ensure Qlib is initialized
        global _qlib_initialized, _qlib_provider_uri
        if not _qlib_initialized:
            _qlib_initialized, _qlib_provider_uri = init_qlib(provider_uri)
            if not _qlib_initialized:
                raise RuntimeError("Qlib initialization failed, please check data path configuration")
        
        # Calculate IC
        result = calculate_ic(
            formula=formula,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            label_expr=label_expr,
            provider_uri=provider_uri
        )
        
        # Print results
        print("=" * 80)
        print("IC Calculation Completed")
        print("=" * 80)
        print(f"Factor Expression: {formula}")
        print(f"Stock Pool: {instruments}")
        print(f"Date Range: {start_date} to {end_date}")
        print()
        print("IC Metrics Results:")
        print(f"  IC Mean: {result['metrics']['ic_mean']:.6f}")
        print(f"  IC Std: {result['metrics']['ic_std']:.6f}")
        print(f"  Information Ratio (IR): {result['metrics']['ir']:.6f}")
        print(f"  Rank IC Mean: {result['metrics']['rank_ic_mean']:.6f}")
        print(f"  Rank IC Std: {result['metrics']['rank_ic_std']:.6f}")
        print(f"  Rank IR: {result['metrics']['rank_ir']:.6f}")
        print(f"  IC Win Rate: {result['metrics']['ic_win_rate']:.4f}")
        print("=" * 80)
        
        logger.info("IC calculation completed")
        
    except Exception as e:
        error_msg = f"IC calculation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"Error: {error_msg}")
        raise RuntimeError(error_msg)


# ============================================================================
# REST API Endpoints (direct call)
# ============================================================================

@app.post("/calculate_ic", response_model=ICResponse)
async def api_calculate_ic(request: ICRequest):
    """Calculate factor IC metrics (REST API)
    
    Args:
        request: IC calculation request parameters
        
    Returns:
        IC metrics results
    """
    global _qlib_initialized, _qlib_provider_uri
    
    try:
        # Ensure Qlib is initialized
        if not _qlib_initialized:
            _qlib_initialized, _qlib_provider_uri = init_qlib(request.provider_uri)
            if not _qlib_initialized:
                return ICResponse(
                    success=False,
                    message="Qlib initialization failed, please check data path configuration"
                )
        
        # Calculate IC
        result = calculate_ic(
            formula=request.formula,
            instruments=request.instruments,
            start_date=request.start_date,
            end_date=request.end_date,
            label_expr=request.label_expr,
            provider_uri=request.provider_uri
        )
        
        return ICResponse(
            success=True,
            metrics=ICMetrics(**result['metrics']),
            message="Calculation successful",
            details=result['details']
        )
        
    except Exception as e:
        logger.error(f"Error calculating IC: {e}", exc_info=True)
        return ICResponse(
            success=False,
            message=f"Calculation failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "qlib_initialized": _qlib_initialized,
        "provider_uri": _qlib_provider_uri,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root path, returns service information"""
    return {
        "service": "MCP IC Calculation Service",
        "version": "1.0.0",
        "description": "MCP protocol based factor IC calculation service",
        "endpoints": {
            "MCP": {
                "GET /tools": "Get available tool list",
                "POST /tools/call": "Call tool"
            },
            "REST API": {
                "POST /calculate_ic": "Calculate factor IC metrics"
            },
            "Other": {
                "GET /health": "Health check",
                "GET /docs": "Swagger documentation",
                "GET /redoc": "ReDoc documentation"
            }
        }
    }


# ============================================================================
# Main Program Entry
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("MCP IC Calculation Service")
    print("=" * 80)
    print()
    print("Service URL: http://localhost:8000")
    print("Swagger Docs: http://localhost:8000/docs")
    print("ReDoc Docs: http://localhost:8000/redoc")
    print()
    print("MCP Endpoints:")
    print("  GET  /tools      - Get available tool list")
    print("  POST /tools/call - Call tool")
    print()
    print("REST API Endpoints:")
    print("  POST /calculate_ic - Calculate factor IC metrics")
    print()
    print("=" * 80)
    
    uvicorn.run(app, host="localhost", port=8000)

