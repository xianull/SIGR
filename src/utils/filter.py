"""
Anti-Leakage Filter for SIGR Framework

Filters LLM-generated descriptions to prevent EXPLICIT label leakage.
Preserves implicit biological information that supports downstream learning.

Key principle:
- KEEP: Factual biological descriptions (implicit signals)
  e.g., "involved in cell cycle regulation", "highly conserved", "hub in PPI network"
- FILTER: Direct predictions and explicit labels
  e.g., "is essential", "is dosage sensitive", "is a marker for X cells"
"""

import re
import logging
from typing import List, Tuple


logger = logging.getLogger(__name__)


# Forbidden patterns - ONLY filter explicit predictions and direct labels
# These patterns indicate direct label leakage, not biological facts
FORBIDDEN_PATTERNS: List[Tuple[str, str]] = [
    # === Predictive/speculative statements (always filter) ===
    (r'\b(likely|probably|possibly)\s+(is|to be|will|would)\s+(a|an)?\s*(essential|sensitive|marker)', 'predictive label'),
    (r'\bpredicted\s+(as|to be)\s+(a|an)?\s*\w+', 'prediction'),
    (r'\bclassified\s+as\s+(a|an)?\s*\w+', 'classification'),
    (r'\bI\s+(predict|believe|think|conclude)\b', 'speculation'),

    # === Direct label declarations (filter explicit labels only) ===
    # Dosage sensitivity - explicit labels only
    (r'\b(is|are)\s+(a\s+)?(haploinsufficient|haplosufficient)\s*(gene)?\b', 'explicit dosage label'),
    (r'\b(is|are)\s+(a\s+)?dosage[- ]?(sensitive|insensitive)\b', 'explicit dosage label'),
    (r'\b(is|are)\s+(a\s+|an\s+)?(essential|non-?essential)\s+gene\b', 'explicit essentiality label'),
    (r'\bthis\s+(gene\s+)?is\s+(an?\s+)?essential\s+gene\b', 'explicit essentiality'),

    # Cell marker - explicit labels only
    (r'\b(is|are)\s+(a\s+)?(specific\s+)?marker\s+(for|of)\s+\w+\s*(cells?)?\b', 'explicit marker label'),
    (r'\b(is|are)\s+specifically\s+expressed\s+in\s+\w+\s+cells?\b', 'explicit cell specificity'),

    # Gene type - explicit classification only
    (r'\b(is|are)\s+(a\s+)?(known\s+)?(oncogene|tumor\s+suppressor|proto-?oncogene)\b', 'explicit gene type'),
    (r'\bthis\s+(is|gene\s+is)\s+(a|an)\s+(kinase|receptor|enzyme|transporter)\b', 'explicit functional type'),

    # Chromatin state - explicit state labels only
    (r'\b(is|are|has)\s+(a\s+)?bivalent\s+(chromatin\s+)?(state|domain|promoter)?\b', 'explicit bivalent label'),
    (r'\b(is|are)\s+(un)?methylated\s+at\s+(the\s+)?promoter\b', 'explicit methylation label'),

    # === Probability/confidence statements ===
    (r'\bprobability\s+of\s+being\b', 'probability statement'),
    (r'\b\d+(\.\d+)?%\s+(chance|probability|confidence)\b', 'percentage prediction'),
    # Removed: confidence score rule was too aggressive and filtered database scores

    # === Future tense predictions ===
    (r'\bwill\s+(likely\s+)?(be|interact|bind|cause)\b', 'future prediction'),
    (r'\bwould\s+(likely\s+)?(be|interact|bind|cause)\b', 'conditional prediction'),

    # === Direct answers to task questions ===
    (r'\b(yes|no),?\s+(this\s+gene\s+)?(is|does|will|can)\b', 'direct answer'),
    (r'\b(positive|negative)\s+(for|sample|case)\b', 'sample classification'),
]

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

    # Functional descriptions
    (r'\bkinase\s+activity\b', 'enzymatic activity - OK'),
    (r'\bbinding\s+(domain|site|capacity)\b', 'structural feature - OK'),
    (r'\bsignaling\s+pathway\b', 'pathway - OK'),
]


def filter_description(description: str, strict: bool = True) -> str:
    """
    Filter LLM-generated description to remove EXPLICIT label leakage.

    Preserves implicit biological information while removing direct predictions.

    Args:
        description: Raw LLM-generated description
        strict: If True, replace forbidden patterns; if False, just log warnings

    Returns:
        Filtered description
    """
    if not description:
        return ""

    filtered = description
    replacements_made = 0

    for pattern, reason in FORBIDDEN_PATTERNS:
        matches = list(re.finditer(pattern, filtered, re.IGNORECASE))

        if matches:
            if strict:
                filtered = re.sub(pattern, '[REDACTED]', filtered, flags=re.IGNORECASE)
                replacements_made += len(matches)
                logger.warning(f"Filtered explicit label: '{reason}' - {matches[0].group()}")
            else:
                logger.warning(f"Found potential leakage ({reason}): {matches[0].group()}")

    if replacements_made > 0:
        logger.info(f"Removed {replacements_made} explicit label(s) from description")

    # Clean up multiple [REDACTED] markers
    filtered = re.sub(r'(\[REDACTED\]\s*)+', '', filtered)

    # Remove empty sentences
    filtered = re.sub(r'\.\s*\.', '.', filtered)
    filtered = re.sub(r'\s+', ' ', filtered)

    return filtered.strip()


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


def sanitize_for_task(description: str, task_name: str) -> str:
    """
    Apply task-specific sanitization.

    Only filters EXPLICIT labels for the specific task.
    Preserves implicit biological information.

    Args:
        description: Description to sanitize
        task_name: Name of the downstream task

    Returns:
        Sanitized description
    """
    # First apply general filtering
    sanitized = filter_description(description)

    # Task-specific EXPLICIT patterns to filter
    # These are direct answers to the task question
    task_explicit_patterns = {
        'ppi': [
            (r'\b(does|will)\s+(not\s+)?interact\s+with\b', 'explicit PPI answer'),
        ],
        'ggi': [
            (r'\b(does|will)\s+(not\s+)?interact\s+with\b', 'explicit GGI answer'),
        ],
        'geneattribute_dosage_sensitivity': [
            (r'\bthis\s+gene\s+is\s+(not\s+)?(dosage\s+)?sensitive\b', 'explicit dosage answer'),
        ],
        'geneattribute_bivalent': [
            (r'\bthis\s+gene\s+(has|is)\s+(not\s+)?bivalent\b', 'explicit bivalent answer'),
        ],
        'cell': [
            (r'\bthis\s+gene\s+is\s+(a\s+)?marker\s+for\b', 'explicit marker answer'),
        ],
    }

    if task_name in task_explicit_patterns:
        for pattern, reason in task_explicit_patterns[task_name]:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
                logger.warning(f"Removed task-specific explicit label: {reason}")

    # Log implicit signals for analysis (not filtered)
    signals = check_implicit_signals(sanitized)
    if signals:
        logger.debug(f"Implicit signals preserved: {signals}")

    return sanitized


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
