"""
Anti-Leakage Filter for SIGR Framework

Filters LLM-generated descriptions to prevent EXPLICIT label leakage.
Preserves implicit biological information that supports downstream learning.

Key principle:
- KEEP: Factual biological descriptions (implicit signals)
  e.g., "involved in cell cycle regulation", "highly conserved", "hub in PPI network"
- FILTER: Direct predictions and explicit labels ONLY
  e.g., "is essential", "is dosage sensitive", "is a marker for X cells"
"""

import re
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filtering a description."""
    original: str  # Original description
    filtered: str  # Filtered description
    changes_made: int  # Number of patterns filtered
    filtered_patterns: List[str]  # List of filtered pattern types


# Forbidden patterns - ONLY filter explicit labels that directly reveal the answer
# These patterns indicate direct label leakage
FORBIDDEN_PATTERNS: List[Tuple[str, str]] = [
    # === Explicit dosage sensitivity labels ===
    (r'\b(is|are)\s+(a\s+)?(haploinsufficient|haplosufficient)\s*(gene)?\b', 'explicit dosage label'),
    (r'\b(is|are)\s+(a\s+)?dosage[- ]?(sensitive|insensitive)\b', 'explicit dosage label'),
    (r'\bthis\s+gene\s+is\s+(not\s+)?(dosage[- ]?)?(sensitive|insensitive)\b', 'explicit dosage answer'),

    # === Explicit essentiality labels ===
    (r'\b(is|are)\s+(a\s+|an\s+)?(essential|non-?essential)\s+gene\b', 'explicit essentiality label'),
    (r'\bthis\s+(gene\s+)?is\s+(an?\s+)?essential\b', 'explicit essentiality'),
    (r'\bthis\s+gene\s+is\s+(not\s+)?essential\b', 'explicit essentiality answer'),

    # === Explicit cell marker labels ===
    (r'\b(is|are)\s+(a\s+)?(specific\s+)?marker\s+(for|of)\s+\w+\s*(cells?)?\b', 'explicit marker label'),
    (r'\bthis\s+gene\s+is\s+(a\s+)?marker\s+for\b', 'explicit marker answer'),

    # === Explicit bivalent/chromatin state labels ===
    (r'\b(is|are|has)\s+(a\s+)?bivalent\s+(chromatin\s+)?(state|domain|promoter)?\b', 'explicit bivalent label'),
    (r'\bthis\s+gene\s+(has|is)\s+(not\s+)?bivalent\b', 'explicit bivalent answer'),

    # === Explicit gene type classifications (oncogene/tumor suppressor) ===
    (r'\b(is|are)\s+(a\s+)?(known\s+)?(oncogene|tumor\s+suppressor)\b', 'explicit gene type'),

    # === Direct predictions with "I predict/believe" ===
    (r'\bI\s+(predict|believe|think|conclude)\s+(that\s+)?(this|it|the)\b', 'speculation'),

    # === Direct yes/no answers ===
    (r'\b(yes|no),?\s+(this\s+gene\s+)?(is|does|will|can)\b', 'direct answer'),
]


def filter_description(description: str, strict: bool = True) -> FilterResult:
    """
    Filter LLM-generated description to remove EXPLICIT label leakage.

    Preserves implicit biological information while removing direct predictions.

    Args:
        description: Raw LLM-generated description
        strict: If True, replace forbidden patterns; if False, just log warnings

    Returns:
        FilterResult with original, filtered description and change details
    """
    if not description:
        return FilterResult(
            original="",
            filtered="",
            changes_made=0,
            filtered_patterns=[]
        )

    filtered = description
    replacements_made = 0
    filtered_patterns = []

    for pattern, reason in FORBIDDEN_PATTERNS:
        matches = list(re.finditer(pattern, filtered, re.IGNORECASE))

        if matches:
            if strict:
                filtered = re.sub(pattern, '', filtered, flags=re.IGNORECASE)
                replacements_made += len(matches)
                filtered_patterns.append(reason)
                logger.warning(f"Filtered explicit label: '{reason}' - {matches[0].group()}")
            else:
                logger.warning(f"Found potential leakage ({reason}): {matches[0].group()}")

    if replacements_made > 0:
        logger.info(f"Removed {replacements_made} explicit label(s) from description")

    # Clean up artifacts from removal
    filtered = re.sub(r'\s+', ' ', filtered)  # Multiple spaces
    filtered = re.sub(r'\.\s*\.', '.', filtered)  # Double periods
    filtered = re.sub(r',\s*,', ',', filtered)  # Double commas
    filtered = re.sub(r'^\s*[,\.]\s*', '', filtered)  # Leading punctuation
    filtered = re.sub(r'\s*[,\.]\s*$', '.', filtered)  # Clean trailing

    return FilterResult(
        original=description,
        filtered=filtered.strip(),
        changes_made=replacements_made,
        filtered_patterns=filtered_patterns
    )


def filter_description_simple(description: str, strict: bool = True) -> str:
    """
    Simple version that returns only the filtered string.

    For backward compatibility.

    Args:
        description: Raw description
        strict: Whether to apply strict filtering

    Returns:
        Filtered description string
    """
    result = filter_description(description, strict)
    return result.filtered


def validate_description(description: str) -> Tuple[bool, List[str]]:
    """
    Validate that a description doesn't contain explicit label leakage.

    Args:
        description: Description to validate

    Returns:
        Tuple of (is_valid, list of issues found)
    """
    issues = []

    for pattern, reason in FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, description, re.IGNORECASE)
        if matches:
            issues.append(f"{reason}: {matches}")

    is_valid = len(issues) == 0

    if not is_valid:
        logger.warning(f"Description has {len(issues)} explicit label(s)")

    return is_valid, issues


# Patterns that are ACCEPTABLE - biological facts (implicit signals)
# These should NOT be filtered as they provide useful implicit information
ACCEPTABLE_PATTERNS: List[Tuple[str, str]] = [
    # Biological process involvement (implicit for many tasks)
    (r'\binvolved\s+in\s+\w+', 'biological process - OK'),
    (r'\bparticipates\s+in\s+\w+', 'pathway participation - OK'),
    (r'\bregulates\s+\w+', 'regulatory role - OK'),

    # Network properties (implicit for essentiality/dosage)
    (r'\bhub\s+(gene|protein|node)\b', 'network centrality - OK'),
    (r'\bhighly\s+connected\b', 'connectivity - OK'),
    (r'\bcentral\s+(role|position)\b', 'centrality - OK'),

    # Conservation (implicit for essentiality)
    (r'\bhighly\s+conserved\b', 'conservation - OK'),
    (r'\bevolutionarily\s+conserved\b', 'conservation - OK'),

    # Expression patterns (implicit for cell type)
    (r'\bexpressed\s+in\s+\w+', 'expression pattern - OK'),
    (r'\bhigh(ly)?\s+express(ed|ion)\b', 'expression level - OK'),
    (r'\benriched\s+in\s+\w+', 'enrichment - OK'),

    # Chromatin/epigenetic features (implicit for bivalent tasks)
    (r'\bH3K\d+me\d\b', 'histone mark - OK'),
    (r'\bchromatin\s+accessibility\b', 'chromatin state - OK'),
    (r'\bpromoter\s+region\b', 'genomic feature - OK'),

    # Functional descriptions (always OK)
    (r'\bkinase\s+activity\b', 'enzymatic activity - OK'),
    (r'\bbinding\s+(domain|site|capacity)\b', 'structural feature - OK'),
    (r'\bsignaling\s+pathway\b', 'pathway - OK'),
    (r'\bclassified\s+as\s+(a\s+)?\w+', 'functional classification - OK'),
    (r'\bprobability\b', 'statistical term - OK'),
]


def check_implicit_signals(description: str) -> List[str]:
    """
    Check what implicit signals are present in the description.

    This is for analysis purposes - these patterns are GOOD and should be kept.

    Args:
        description: Description to analyze

    Returns:
        List of implicit signals found
    """
    signals = []

    for pattern, signal_type in ACCEPTABLE_PATTERNS:
        if re.search(pattern, description, re.IGNORECASE):
            signals.append(signal_type)

    return signals


def sanitize_for_task(description: str, task_name: str) -> FilterResult:
    """
    Apply task-specific sanitization.

    Only filters EXPLICIT labels for the specific task.
    Preserves implicit biological information.

    Args:
        description: Description to sanitize
        task_name: Name of the downstream task

    Returns:
        FilterResult with original and sanitized description
    """
    # First apply general filtering
    result = filter_description(description)
    sanitized = result.filtered

    # Task-specific EXPLICIT patterns to filter
    task_explicit_patterns = {
        'ppi': [
            (r'\b(does|will)\s+(not\s+)?interact\s+with\b', 'explicit PPI answer'),
        ],
        'ggi': [
            (r'\b(does|will)\s+(not\s+)?interact\s+with\b', 'explicit GGI answer'),
        ],
        'geneattribute_dosage_sensitivity': [
            # Already covered in general patterns
        ],
        'geneattribute_bivalent': [
            # Already covered in general patterns
        ],
        'cell': [
            # Already covered in general patterns
        ],
    }

    additional_filtered = []
    if task_name in task_explicit_patterns:
        for pattern, reason in task_explicit_patterns[task_name]:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
                additional_filtered.append(reason)
                logger.warning(f"Removed task-specific explicit label: {reason}")

    # Log implicit signals for analysis (not filtered)
    signals = check_implicit_signals(sanitized)
    if signals:
        logger.debug(f"Implicit signals preserved: {signals}")

    return FilterResult(
        original=description,
        filtered=sanitized.strip(),
        changes_made=result.changes_made + len(additional_filtered),
        filtered_patterns=result.filtered_patterns + additional_filtered
    )


def is_safe_description(description: str) -> bool:
    """
    Quick check if description is safe (no explicit leakage).

    Args:
        description: Description to check

    Returns:
        True if safe, False if explicit leakage detected
    """
    is_valid, _ = validate_description(description)
    return is_valid
