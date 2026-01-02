"""
Utility Functions
=================
Helper functions for logging, validation, token counting, cost calculation, etc.

Functions:
- Logging setup
- Decision ID generation
- Token counting
- Cost estimation
- Data validation
- Formatting helpers
"""

import logging
import sys
import uuid
import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from config.settings import settings


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level: str = None, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    log_level = log_level or settings.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# ID GENERATION
# ============================================================================

def generate_decision_id() -> str:
    """
    Generate unique decision ID.
    
    Returns:
        UUID-based decision ID
    """
    return f"decision_{uuid.uuid4().hex[:16]}"


def generate_analysis_id() -> str:
    """
    Generate unique analysis ID.
    
    Returns:
        UUID-based analysis ID
    """
    return f"analysis_{uuid.uuid4().hex[:16]}"


def generate_hash(data: Any) -> str:
    """
    Generate SHA-256 hash of data.
    
    Args:
        data: Data to hash (string, dict, or any JSON-serializable object)
        
    Returns:
        Hex digest of hash
    """
    if isinstance(data, str):
        hash_input = data.encode()
    elif isinstance(data, dict):
        hash_input = json.dumps(data, sort_keys=True).encode()
    else:
        hash_input = str(data).encode()
    
    return hashlib.sha256(hash_input).hexdigest()


# ============================================================================
# TOKEN COUNTING & COST ESTIMATION
# ============================================================================

def estimate_token_count(text: str) -> int:
    """
    Estimate token count for text.
    
    Note: This is a rough approximation. Actual token count varies by model.
    Rule of thumb: ~4 characters per token for English text.
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    # Simple estimation: 4 chars = 1 token
    return len(text) // 4


def calculate_estimated_cost(total_tokens: int, model_names: List[str]) -> float:
    """
    Calculate estimated API cost based on tokens and models used.
    
    Pricing (as of 2024):
    - Groq Llama 3.3 70B: $0.59/1M tokens (input + output)
    - Groq Qwen 2.5 72B: $0.59/1M tokens (input + output)
    - Google Gemini 2.0 Flash: $0.075/1M input, $0.30/1M output (avg ~$0.19/1M)
    
    Args:
        total_tokens: Total tokens used
        model_names: List of model names used
        
    Returns:
        Estimated cost in USD
    """
    # Model pricing per 1M tokens (averaged input/output)
    pricing = {
        "llama-3.3-70b-versatile": 0.00000059,  # $0.59/1M
        "qwen-2.5-72b-instruct": 0.00000059,    # $0.59/1M
        "gemini-2.0-flash-exp": 0.00000019,     # ~$0.19/1M (avg)
        "gemini-2.0-flash": 0.00000019,
        "default": 0.00000050  # Default fallback
    }
    
    # Calculate average cost per token across models used
    total_cost_per_token = 0.0
    for model in model_names:
        # Find matching pricing
        cost = pricing.get(model, pricing["default"])
        total_cost_per_token += cost
    
    # Average across models
    avg_cost_per_token = total_cost_per_token / len(model_names) if model_names else pricing["default"]
    
    # Calculate total cost
    total_cost = total_tokens * avg_cost_per_token
    
    return round(total_cost, 6)


def calculate_cost_breakdown(agent_responses: List[Any]) -> Dict[str, float]:
    """
    Calculate cost breakdown by agent.
    
    Args:
        agent_responses: List of AgentResponse objects
        
    Returns:
        Dictionary with cost per agent
    """
    pricing = {
        "llama-3.3-70b-versatile": 0.00000059,
        "qwen-2.5-72b-instruct": 0.00000059,
        "gemini-2.0-flash-exp": 0.00000019,
    }
    
    breakdown = {}
    for response in agent_responses:
        model = response.model_name
        tokens = response.tokens_used
        cost_per_token = pricing.get(model, 0.00000050)
        breakdown[response.agent_name] = round(tokens * cost_per_token, 6)
    
    return breakdown


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_risk_score(score: float) -> bool:
    """
    Validate risk score is in valid range.
    
    Args:
        score: Risk score
        
    Returns:
        True if valid
    """
    return 0 <= score <= 100


def validate_confidence(confidence: float) -> bool:
    """
    Validate confidence is in valid range.
    
    Args:
        confidence: Confidence value
        
    Returns:
        True if valid
    """
    return 0 <= confidence <= 100


def validate_verdict(verdict: str) -> bool:
    """
    Validate verdict is one of allowed values.
    
    Args:
        verdict: Verdict string
        
    Returns:
        True if valid
    """
    allowed = ["malicious", "benign", "uncertain", "error"]
    return verdict in allowed


def validate_threat_level(threat_level: str) -> bool:
    """
    Validate threat level is one of allowed values.
    
    Args:
        threat_level: Threat level string
        
    Returns:
        True if valid
    """
    allowed = ["critical", "high", "medium", "low", "unknown"]
    return threat_level in allowed


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_timestamp(dt: datetime = None) -> str:
    """
    Format datetime as ISO string.
    
    Args:
        dt: Datetime object (default: now)
        
    Returns:
        ISO formatted string
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat() + "Z"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2.5s", "1m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.0f}s"


def format_cost(cost_usd: float) -> str:
    """
    Format cost in USD.
    
    Args:
        cost_usd: Cost in USD
        
    Returns:
        Formatted string (e.g., "$0.0123")
    """
    return f"${cost_usd:.4f}"


def format_tokens(token_count: int) -> str:
    """
    Format token count with thousand separators.
    
    Args:
        token_count: Number of tokens
        
    Returns:
        Formatted string (e.g., "12,345")
    """
    return f"{token_count:,}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# DATA PROCESSING HELPERS
# ============================================================================

def sanitize_package_name(name: str) -> str:
    """
    Sanitize package name for safe processing.
    
    Args:
        name: Package name
        
    Returns:
        Sanitized name
    """
    # Remove potentially dangerous characters
    import re
    sanitized = re.sub(r'[^\w\-\.]', '', name)
    return sanitized.lower()


def extract_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL.
    
    Args:
        url: URL string
        
    Returns:
        Domain or None if invalid
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or parsed.path.split('/')[0]
    except Exception:
        return None


def parse_version(version: str) -> Dict[str, int]:
    """
    Parse semantic version string.
    
    Args:
        version: Version string (e.g., "1.2.3")
        
    Returns:
        Dictionary with major, minor, patch
    """
    try:
        parts = version.split('.')
        return {
            "major": int(parts[0]) if len(parts) > 0 else 0,
            "minor": int(parts[1]) if len(parts) > 1 else 0,
            "patch": int(parts[2]) if len(parts) > 2 else 0
        }
    except (ValueError, IndexError):
        return {"major": 0, "minor": 0, "patch": 0}


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


# ============================================================================
# FILE HELPERS
# ============================================================================

def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_file(path: Path) -> Dict:
    """
    Read JSON file.
    
    Args:
        path: File path
        
    Returns:
        Parsed JSON as dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(path: Path, data: Dict, indent: int = 2):
    """
    Write dictionary to JSON file.
    
    Args:
        path: File path
        data: Data to write
        indent: JSON indentation
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


# ============================================================================
# SECURITY HELPERS
# ============================================================================

def is_suspicious_string(text: str) -> bool:
    """
    Check if string contains suspicious patterns.
    
    Args:
        text: Text to check
        
    Returns:
        True if suspicious
    """
    suspicious_patterns = [
        'eval(',
        'exec(',
        'atob(',
        'btoa(',
        'Function(',
        '__import__',
        'subprocess',
        'child_process',
        '/bin/sh',
        '/bin/bash',
        'rm -rf',
        'curl | bash',
        'wget | sh'
    ]
    
    text_lower = text.lower()
    return any(pattern.lower() in text_lower for pattern in suspicious_patterns)


def calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Entropy value (0-8, higher = more random)
    """
    if not text:
        return 0.0
    
    import math
    from collections import Counter
    
    # Count character frequencies
    counter = Counter(text)
    length = len(text)
    
    # Calculate entropy
    entropy = 0.0
    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    
    return entropy


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_decision_summary(decision: Any):
    """
    Pretty print decision summary.
    
    Args:
        decision: CouncilDecision object
    """
    print("\n" + "=" * 80)
    print(f"COUNCIL DECISION SUMMARY")
    print("=" * 80)
    print(f"Package: {decision.package_name}@{decision.package_version}")
    print(f"Verdict: {decision.verdict.upper()}")
    print(f"Threat Level: {decision.threat_level.upper()}")
    print(f"Risk Score: {decision.final_risk_score}/100")
    print(f"Confidence: {decision.final_confidence}%")
    print(f"\nAnalysis Time: {format_duration(decision.analysis_duration_seconds)}")
    print(f"Tokens Used: {format_tokens(decision.total_tokens_used)}")
    print(f"Estimated Cost: {format_cost(decision.estimated_cost_usd)}")
    
    if decision.recommended_actions:
        print(f"\nRecommended Actions:")
        for i, action in enumerate(decision.recommended_actions[:5], 1):
            print(f"  {i}. {action}")
    
    if decision.requires_human_review:
        print(f"\n⚠️  REQUIRES HUMAN REVIEW")
    
    print("=" * 80 + "\n")


def print_agent_responses(responses: List[Any]):
    """
    Pretty print agent responses.
    
    Args:
        responses: List of AgentResponse objects
    """
    print("\n" + "-" * 80)
    print("AGENT RESPONSES")
    print("-" * 80)
    
    for response in responses:
        print(f"\n{response.agent_name.upper()} ({response.model_name})")
        print(f"  Risk: {response.risk_score}/100")
        print(f"  Confidence: {response.confidence}%")
        print(f"  Verdict: {response.verdict}")
        print(f"  Tokens: {response.tokens_used}")
        print(f"  Explanation: {truncate_text(response.explanation, 150)}")
    
    print("-" * 80 + "\n")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    
    # ID Generation
    'generate_decision_id',
    'generate_analysis_id',
    'generate_hash',
    
    # Token & Cost
    'estimate_token_count',
    'calculate_estimated_cost',
    'calculate_cost_breakdown',
    
    # Validation
    'validate_risk_score',
    'validate_confidence',
    'validate_verdict',
    'validate_threat_level',
    
    # Formatting
    'format_timestamp',
    'format_duration',
    'format_cost',
    'format_tokens',
    'truncate_text',
    
    # Data Processing
    'sanitize_package_name',
    'extract_domain',
    'parse_version',
    'merge_dicts',
    
    # File Operations
    'ensure_directory',
    'read_json_file',
    'write_json_file',
    
    # Security
    'is_suspicious_string',
    'calculate_entropy',
    
    # Pretty Printing
    'print_decision_summary',
    'print_agent_responses'
]