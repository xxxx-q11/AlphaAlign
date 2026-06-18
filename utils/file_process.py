
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import json



def explore_repo_structure(repo_path: str) -> dict:

    
    # 1. Generate simplified directory tree (max 3 levels, ignore node_modules/.git etc.)
    tree_lines = []
    for root, dirs, files in os.walk(repo_path):
        level = root.replace(str(repo_path), "").count(os.sep)
        if level > 3: 
            continue
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", ".venv"}]
        indent = "│   " * level
        tree_lines.append(f"{indent}├── {os.path.basename(root)}/")
        for f in sorted(files):
            tree_lines.append(f"{indent}│   ├── {f}")
    
    file_tree = "\n".join(tree_lines[:100])  # Truncate to prevent overflow
    
    # # 2. Read README
    # readme_path = repo_path / "README.md"
    # readme = readme_path.read_text() if readme_path.exists() else ""
    
    return {
        "file_tree": file_tree,
        #readme_content": readme[:2000]  # Truncate
    }



def find_training_scripts(repo_structure: Dict[str, Any], repo_path: str) -> List[str]:
    """
    Find training scripts from explore_repo_structure output (only search first-level directories)
    
    Args:
        repo_structure: Dictionary returned by explore_repo_structure function, containing "file_tree" key
        repo_path: Root path of the repository, used to build complete file paths
        
    Returns:
        List of matched training script paths (absolute paths), only including scripts in first-level directories
        
    Example:
        >>> structure = explore_repo_structure("/path/to/repo")
        >>> scripts = find_training_scripts(structure, "/path/to/repo")
        >>> # Returns: ["/path/to/repo/workspace/AlphaForge/train_GP.py", ...]
    """
    file_tree = repo_structure.get("file_tree", "")
    if not file_tree:
        return []
    
    # Match patterns: train_*.py and *train*.py
    patterns = [
        r"train_.*\.py$",  # train_*.py
        r".*train.*\.py$",  # *train*.py
    ]
    
    # Compile regular expressions
    compiled_patterns = [re.compile(pattern) for pattern in patterns]
    
    # Parse directory tree, rebuild file paths
    training_scripts = []
    path_stack = []  # Track current path level
    
    for line in file_tree.split("\n"):
        if not line.strip():
            continue
        
        # Calculate indentation level (based on number of "│   ")
        # Each occurrence of "│   " represents one level of indentation
        stripped = line.lstrip()
        indent_level = (len(line) - len(stripped)) // 4
        
        # Only process first-level directories (indent_level <= 1)
        # indent_level = 0: root directory (skip, as repo_path is already complete path)
        # indent_level = 1: first-level subdirectory
        if indent_level > 1:
            continue
        
        # Remove all tree structure symbols, extract actual content
        content = re.sub(r"^[├│└──\s]+", "", stripped).strip()
        if not content:
            continue
        
        if content.endswith("/"):
            # This is a directory
            # Skip root directory (indent_level == 0), as repo_path already contains root directory
            if indent_level == 0:
                continue
                
            dir_name = content.rstrip("/")
            # Adjust path stack to current level (when indent_level=1, path_stack should only contain current directory)
            path_stack = path_stack[:indent_level - 1]  # When indent_level=1, path_stack[:0] is empty list
            path_stack.append(dir_name)
        elif content.endswith(".py"):
            # This is a Python file
            # Process files in root directory (indent_level=0, path_stack is empty)
            # or files in first-level subdirectories (indent_level=1, path_stack length is 1)
            if indent_level == 0:
                # File in root directory, path_stack should be empty
                file_name = content
                # Check if matches training script pattern
                is_training_script = False
                for pattern in compiled_patterns:
                    if pattern.match(file_name):
                        is_training_script = True
                        break
                
                if is_training_script:
                    # Build complete path (file in root directory)
                    full_path = os.path.join(repo_path, file_name)
                    # Normalize path
                    full_path = os.path.normpath(full_path)
                    training_scripts.append(full_path)
            elif indent_level == 1 and len(path_stack) == 1:
                # File in first-level subdirectory
                file_name = content
                # Check if matches training script pattern
                is_training_script = False
                for pattern in compiled_patterns:
                    if pattern.match(file_name):
                        is_training_script = True
                        break
                
                if is_training_script:
                    # Build complete path
                    relative_path = os.path.join(*path_stack, file_name)
                    # Build absolute path
                    full_path = os.path.join(repo_path, relative_path)
                    # Normalize path
                    full_path = os.path.normpath(full_path)
                    training_scripts.append(full_path)
    
    # Remove duplicates and sort
    training_scripts = sorted(list(set(training_scripts)))
    
    return training_scripts


def find_readme_files(repo_structure: Dict[str, Any], repo_path: str) -> List[str]:
    """
    Find README documents from explore_repo_structure output (only search first-level directories)
    
    Args:
        repo_structure: Dictionary returned by explore_repo_structure function, containing "file_tree" key
        repo_path: Root path of the repository, used to build complete file paths
        
    Returns:
        List of matched README file paths (absolute paths), only including README files in first-level directories
        
    Example:
        >>> structure = explore_repo_structure("/path/to/repo")
        >>> readme_files = find_readme_files(structure, "/path/to/repo")
        >>> # Returns: ["/path/to/repo/workspace/AlphaForge/README.md", ...]
    """
    file_tree = repo_structure.get("file_tree", "")
    if not file_tree:
        return []
    
    # Match patterns: README-related file names (case-insensitive)
    # Supports: README, README.md, README.txt, README.rst, readme.md, etc.
    patterns = [
        r"^README$",           # README (no extension)
        r"^README\.(md|txt|rst|markdown)$",  # README.md, README.txt, etc.
    ]
    
    # Compile regular expressions (case-insensitive)
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    # Parse directory tree, rebuild file paths
    readme_files = []
    path_stack = []  # Track current path level
    
    for line in file_tree.split("\n"):
        if not line.strip():
            continue
        
        # Calculate indentation level (based on number of "│   ")
        # Each occurrence of "│   " represents one level of indentation
        stripped = line.lstrip()
        indent_level = (len(line) - len(stripped)) // 4
        
        # Only process first-level directories (indent_level <= 1)
        # indent_level = 0: root directory (skip directory itself, but process files)
        # indent_level = 1: first-level subdirectory
        if indent_level > 1:
            continue
        
        # Remove all tree structure symbols, extract actual content
        content = re.sub(r"^[├│└──\s]+", "", stripped).strip()
        if not content:
            continue
        
        if content.endswith("/"):
            # This is a directory
            # Skip root directory (indent_level == 0), as repo_path already contains root directory
            if indent_level == 0:
                continue
                
            dir_name = content.rstrip("/")
            # Adjust path stack to current level (when indent_level=1, path_stack should only contain current directory)
            path_stack = path_stack[:indent_level - 1]  # When indent_level=1, path_stack[:0] is empty list
            path_stack.append(dir_name)
        else:
            # This is a file (possibly README)
            # Process files in root directory (indent_level=0, path_stack is empty)
            # or files in first-level subdirectories (indent_level=1, path_stack length is 1)
            if indent_level == 0:
                # File in root directory, path_stack should be empty
                file_name = content
                # Check if matches README file pattern
                is_readme = False
                for pattern in compiled_patterns:
                    if pattern.match(file_name):
                        is_readme = True
                        break
                
                if is_readme:
                    # Build complete path (file in root directory)
                    full_path = os.path.join(repo_path, file_name)
                    # Normalize path
                    full_path = os.path.normpath(full_path)
                    readme_files.append(full_path)
            elif indent_level == 1 and len(path_stack) == 1:
                # File in first-level subdirectory
                file_name = content
                # Check if matches README file pattern
                is_readme = False
                for pattern in compiled_patterns:
                    if pattern.match(file_name):
                        is_readme = True
                        break
                
                if is_readme:
                    # Build complete path
                    relative_path = os.path.join(*path_stack, file_name)
                    # Build absolute path
                    full_path = os.path.join(repo_path, relative_path)
                    # Normalize path
                    full_path = os.path.normpath(full_path)
                    readme_files.append(full_path)
    
    # Remove duplicates and sort
    readme_files = sorted(list(set(readme_files)))
    
    return readme_files


def read_file_for_llm(file_path: str, encoding: str = "utf-8", max_size: Optional[int] = 10000) -> str:
    """
    Read file content as input for LLM
    
    Args:
        file_path: File path (absolute or relative path)
        encoding: File encoding, defaults to utf-8. If reading fails, will try other common encodings
        max_size: Maximum read size (in bytes), files exceeding this size will be truncated. Defaults to 10000 bytes. Pass None for no limit
        
    Returns:
        File content string
        
    Raises:
        FileNotFoundError: File does not exist
        IOError: Error occurred while reading file
        
    Example:
        >>> content = read_file_for_llm("/path/to/script.py")
        >>> # Returns file content string, can be directly used as LLM input
    """
    #file_path = os.path.normpath(file_path)
    print(f"Reading file: {file_path}")
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    if not os.path.isfile(file_path):
        raise IOError(f"Path is not a file: {file_path}")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    if max_size and file_size > max_size:
        # If file is too large, only read first max_size bytes
        with open(file_path, 'rb') as f:
            content_bytes = f.read(max_size)
        # Try to decode
        try:
            content = content_bytes.decode(encoding)
            content += f"\n\n[File truncated, original size: {file_size} bytes, read: {max_size} bytes]"
        except UnicodeDecodeError:
            # If specified encoding fails, try other encodings
            for alt_encoding in ['utf-8', 'gbk', 'latin-1', 'cp1252']:
                try:
                    content = content_bytes.decode(alt_encoding)
                    content += f"\n\n[File truncated, original size: {file_size} bytes, read: {max_size} bytes]"
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise IOError(f"Unable to decode file content, tried multiple encodings: {file_path}")
    else:
        # Normal read entire file
        encodings_to_try = [encoding, 'utf-8', 'gbk', 'latin-1', 'cp1252']
        content = None
        
        for enc in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise IOError(f"Error occurred while reading file: {e}")
        
        if content is None:
            raise IOError(f"Unable to read file, tried multiple encodings: {file_path}")
    
    return content


def select_training_script(script_paths: List[str]) -> str:
    """
    Select a script from training script list
    
    Args:
        script_paths: List of script paths returned by find_training_scripts function
        
    Returns:
        User-selected script path
        
    Raises:
        ValueError: If list is empty or user input is invalid
        
    Example:
        >>> scripts = find_training_scripts(structure, "/path/to/repo")
        >>> selected = select_training_script(scripts)
        >>> # Display list, return selected path after user selection
    """
    if not script_paths:
        raise ValueError("Script list is empty, cannot select")
    
    # Display script list
    print("\nFound the following training scripts:")
    print("-" * 80)
    for idx, script_path in enumerate(script_paths, start=1):
        # Display relative path or filename for better readability
        display_path = script_path
        if len(script_path) > 70:
            # If path is too long, only show last part
            display_path = "..." + script_path[-67:]
        print(f"  [{idx}] {display_path}")
    print("-" * 80)
    
    # Get user input
    while True:
        try:
            choice = input(f"\nPlease select a script (1-{len(script_paths)}), or enter 'q' to quit: ").strip()
            
            # Allow quit
            if choice.lower() == 'q':
                raise ValueError("User cancelled selection")
            
            # Convert to integer
            choice_num = int(choice)
            
            # Validate range
            if 1 <= choice_num <= len(script_paths):
                selected_path = script_paths[choice_num - 1]
                print(f"\nSelected: {selected_path}")
                return selected_path
            else:
                print(f"Invalid selection, please enter a number between 1 and {len(script_paths)}")
        except ValueError as e:
            if "User cancelled selection" in str(e):
                raise
            print("Invalid input, please enter a number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nUser interrupted operation")
            raise ValueError("User interrupted selection")
        except Exception as e:
            print(f"Error occurred: {e}")

def get_top_factors_from_gp_json(
    json_file_path: str,
    top_n: int = 50,
    ic_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Filter factors with highest IC values from JSON file saved by GP algorithm
    
    Supports two JSON formats:
    1. Old format: {"cache": {"expression": IC_value, ...}}
    2. New format: {"factors": [{"qlib_expression": "...", "train_ic": ..., "original_expression": "..."}, ...]}
       For new format, prioritize original_expression as the expression
    
    Args:
        json_file_path: JSON file path saved by GP algorithm (absolute or relative path)
        top_n: Return top N factors, defaults to 20
        
    Returns:
        List of dictionaries containing factor expressions and IC values, sorted by IC value in descending order
        Each dictionary format: {"expression": "expression string", "ic": IC_value}
        
    Raises:
        FileNotFoundError: File does not exist
        ValueError: JSON format error or missing cache/factors field
        IOError: Error occurred while reading file
        
    Example:
        >>> factors = get_top_factors_from_gp_json("/path/to/10.json")
        >>> # Returns: [{"expression": "TsMinDiff(...)", "ic": 0.0214}, ...]
        >>> factors = get_top_factors_from_gp_json("/path/to/4_qlib_factors.json")
        >>> # Returns: [{"expression": "TsIr(TsMean(volume,30),50)", "ic": 0.0445}, ...]
    """
    # Check if file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File does not exist: {json_file_path}")
    
    if not os.path.isfile(json_file_path):
        raise IOError(f"Path is not a file: {json_file_path}")
    
    # Read JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON format error: {e}")
    except Exception as e:
        raise IOError(f"Error occurred while reading file: {e}")
    
    factors_list = []
    
    # Support two formats:
    # 1. Old format: {"cache": {"expression": IC_value, ...}}
    # 2. New format: {"factors": [{"qlib_expression": "...", "train_ic": ..., "original_expression": "...", "train_rank_ic": ...}, ...]}
    if "cache" in data:
        # Old format: extract from cache dictionary
        cache = data["cache"]
        if not isinstance(cache, dict):
            raise ValueError("'cache' field must be a dictionary type")
        
        # Old format has no distinction between qlib_expression and original_expression; just use expression for both.
        factors_list = [
            {
                "expression": expr,
                "original_expression": expr,
                "qlib_expression": None,
                "train_ic": ic_value,
                "train_rank_ic": None,
                "valid_ic": None,
                "valid_rank_ic": None,
                "test_ic": None,
                "test_rank_ic": None,
                "ic": ic_value,
                "rank_ic_valid": None,
                "metrics": {
                    "train": {"ic": ic_value, "rank_ic": None, "icir": 0.0, "rank_icir": 0.0},
                    "valid": {"ic": None, "rank_ic": None, "icir": 0.0, "rank_icir": 0.0},
                    "test": {"ic": None, "rank_ic": None, "icir": 0.0, "rank_icir": 0.0},
                },
            }
            for expr, ic_value in cache.items()
        ]
    
    elif "factors" in data:
        # New format: extract from factors array
        factors = data["factors"]
        if not isinstance(factors, list):
            raise ValueError("'factors' field must be an array type")
        
        for factor in factors:
            if not isinstance(factor, dict):
                continue
            
            original_expression = factor.get("original_expression") or factor.get("expression")
            qlib_expression = factor.get("qlib_expression")
            expression = original_expression or qlib_expression
            if not expression:
                continue
            
            metrics = factor.get("metrics", {}) or {}
            train_metrics = metrics.get("train", {}) if isinstance(metrics, dict) else {}
            valid_metrics = metrics.get("valid", {}) if isinstance(metrics, dict) else {}
            test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}

            train_ic_value = factor.get("train_ic")
            if train_ic_value is None:
                train_ic_value = train_metrics.get("ic", factor.get("ic"))

            train_rank_ic_value = factor.get("train_rank_ic")
            if train_rank_ic_value is None:
                train_rank_ic_value = train_metrics.get("rank_ic", factor.get("rank_ic"))

            valid_ic_value = factor.get("valid_ic")
            if valid_ic_value is None:
                valid_ic_value = valid_metrics.get("ic", factor.get("ic_valid"))

            valid_rank_ic_value = factor.get("valid_rank_ic")
            if valid_rank_ic_value is None:
                valid_rank_ic_value = valid_metrics.get("rank_ic", factor.get("rank_ic_valid"))

            test_ic_value = factor.get("test_ic")
            if test_ic_value is None:
                test_ic_value = test_metrics.get("ic", factor.get("ic_test"))

            test_rank_ic_value = factor.get("test_rank_ic")
            if test_rank_ic_value is None:
                test_rank_ic_value = test_metrics.get("rank_ic", factor.get("rank_ic_test"))

            # Get train IC value
            if train_ic_value is None:
                continue

            factors_list.append(
                {
                    "expression": expression,
                    "original_expression": original_expression,
                    "qlib_expression": qlib_expression,
                    "train_ic": train_ic_value,
                    "train_rank_ic": train_rank_ic_value,
                    "valid_ic": valid_ic_value,
                    "valid_rank_ic": valid_rank_ic_value,
                    "test_ic": test_ic_value,
                    "test_rank_ic": test_rank_ic_value,
                    "ic": train_ic_value,
                    "rank_ic_valid": valid_rank_ic_value,
                    "metrics": metrics,
                }
            )
    
    else:
        raise ValueError("JSON file missing 'cache' or 'factors' field, unable to identify file format")
    
    # No longer hardcode thresholds here; let the upstream workflow control it.
    valid_factors = []
    for item in factors_list:
        expr = item.get("expression")
        ic = item.get("train_ic", item.get("ic"))
        if not isinstance(ic, (int, float)):
            continue
        if ic_threshold is not None and ic <= ic_threshold:
            continue
        valid_factors.append(item)
    
    # Sort by validation rank_ic in descending order.
    def get_rank_ic_for_sort(item):
        """Get rank_ic value for sorting"""
        rank_ic = item.get("valid_rank_ic")
        if rank_ic is None:
            return float('-inf')  # None values placed at end
        if isinstance(rank_ic, str):
            try:
                return float(rank_ic)
            except (ValueError, TypeError):
                return float('-inf')
        if isinstance(rank_ic, (int, float)):
            return float(rank_ic)
        return float('-inf')
    
    valid_factors.sort(key=get_rank_ic_for_sort, reverse=True)
    
    # Take top top_n factors
    top_factors = valid_factors[:top_n]
    
    # Convert to dictionary list format
    result = []
    for factor in top_factors:
        expression = factor.get("expression")
        qlib_expression = factor.get("qlib_expression")
        result.append(
            {
                "expression": expression,
                "original_expression": factor.get("original_expression"),
                "qlib_expression": qlib_expression,
                "train_ic": factor.get("train_ic", factor.get("ic")),
                "train_rank_ic": factor.get("train_rank_ic"),
                "valid_ic": factor.get("valid_ic"),
                "valid_rank_ic": factor.get("valid_rank_ic", factor.get("rank_ic_valid")),
                "test_ic": factor.get("test_ic"),
                "test_rank_ic": factor.get("test_rank_ic"),
                "ic": factor.get("train_ic", factor.get("ic")),
                "rank_ic_valid": factor.get("valid_rank_ic", factor.get("rank_ic_valid")),
                "metrics": factor.get("metrics", {}),
            }
        )

    return result
