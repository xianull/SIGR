"""
Strategy Definition for SIGR Actor

Defines the action space for the MDP - the strategy configuration
that the Actor uses to guide KG extraction and text generation.
"""

import json
import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Dict, Any


logger = logging.getLogger(__name__)

SamplingMethod = Literal['top_k', 'random', 'weighted']
DescriptionLength = Literal['short', 'medium', 'long']
DescriptionFocus = Literal['balanced', 'network', 'function', 'phenotype']
ContextWindow = Literal['minimal', 'local', 'extended', 'full']
PromptStyle = Literal['analytical', 'narrative', 'structured', 'comparative']
FeatureSelection = Literal['all', 'essential', 'diverse', 'task_specific']


@dataclass
class StrategyConfig:
    """Configuration for KG extraction and text generation strategy.

    This represents the "Action" in the MDP formulation.

    Parameters:
        edge_types: List of edge types to include in subgraph extraction
        max_hops: Maximum depth of graph traversal (1-3)
        sampling: Sampling method for neighbors (top_k, random, weighted)
        max_neighbors: Global maximum neighbors per edge type

        # Extended parameters
        description_length: Target description length (short=50-100, medium=100-150, long=150-250 words)
        edge_weights: Per-edge-type importance weights for formatting priority
        neighbors_per_type: Fine-grained neighbor limits per edge type
        include_statistics: Whether to include statistical summaries in description
        focus_keywords: Keywords to emphasize in the description

        # New strategy dimensions (v2)
        description_focus: Focus mode for description generation
        context_window: How much KG context to include
        prompt_style: Style/format of the generated description
        feature_selection: How to select features for description
        generation_passes: Number of refinement passes (1-3)

        prompt_template: LLM prompt template
        reasoning: LLM's reasoning for this strategy
    """
    # KG extraction parameters (existing)
    edge_types: List[str] = field(default_factory=lambda: ['PPI', 'GO', 'HPO'])
    max_hops: int = 2
    sampling: SamplingMethod = 'top_k'
    max_neighbors: int = 50

    # Extended parameters (existing)
    description_length: DescriptionLength = 'medium'  # short=50-100, medium=100-150, long=150-250 words
    edge_weights: Dict[str, float] = field(default_factory=dict)  # e.g., {'PPI': 1.0, 'GO': 0.8}
    neighbors_per_type: Dict[str, int] = field(default_factory=dict)  # e.g., {'PPI': 30, 'GO': 20}
    include_statistics: bool = True  # Include counts, scores, etc.
    focus_keywords: List[str] = field(default_factory=list)  # e.g., ['hub', 'conserved']

    # New strategy dimensions (v2)
    description_focus: DescriptionFocus = 'balanced'  # balanced, network, function, phenotype
    context_window: ContextWindow = 'full'  # minimal, local, extended, full
    prompt_style: PromptStyle = 'analytical'  # analytical, narrative, structured, comparative
    feature_selection: FeatureSelection = 'all'  # all, essential, diverse, task_specific
    generation_passes: int = 1  # 1-3 passes for refinement

    # Text generation parameters
    prompt_template: str = ""

    # Metadata
    reasoning: str = ""  # LLM's reasoning for this strategy

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """Create from dictionary."""
        # Filter to valid fields only
        valid_fields = {
            'edge_types', 'max_hops', 'sampling', 'max_neighbors',
            'description_length', 'edge_weights', 'neighbors_per_type',
            'include_statistics', 'focus_keywords',
            'description_focus', 'context_window', 'prompt_style',
            'feature_selection', 'generation_passes',
            'prompt_template', 'reasoning'
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


# Description length word count ranges
DESCRIPTION_LENGTH_WORDS = {
    'short': (50, 100),
    'medium': (100, 150),
    'long': (150, 250),
}


class Strategy:
    """
    Strategy manager for the Actor.

    Handles strategy initialization, validation, and parsing from LLM outputs.
    """

    # Valid edge types in the KG
    VALID_EDGE_TYPES = {
        'PPI', 'GO', 'HPO', 'TRRUST', 'CellMarker', 'Reactome'
    }

    # Valid sampling methods
    VALID_SAMPLING = {'top_k', 'random', 'weighted'}

    # Valid description lengths
    VALID_DESCRIPTION_LENGTHS = {'short', 'medium', 'long'}

    # Parameter constraints
    MIN_HOPS = 1
    MAX_HOPS = 3
    MIN_NEIGHBORS = 10
    MAX_NEIGHBORS = 200

    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize strategy with optional config."""
        self.config = config or StrategyConfig()
        self._validate()

    def _validate(self):
        """Validate strategy configuration."""
        # Validate edge types
        invalid_edges = set(self.config.edge_types) - self.VALID_EDGE_TYPES
        if invalid_edges:
            raise ValueError(f"Invalid edge types: {invalid_edges}")

        # Validate sampling method
        if self.config.sampling not in self.VALID_SAMPLING:
            raise ValueError(f"Invalid sampling method: {self.config.sampling}")

        # Validate description length
        if self.config.description_length not in self.VALID_DESCRIPTION_LENGTHS:
            logger.warning(f"Invalid description_length: {self.config.description_length}, using 'medium'")
            self.config.description_length = 'medium'

        # Validate edge_weights (values should be 0.0-1.0)
        if self.config.edge_weights:
            for edge_type, weight in self.config.edge_weights.items():
                if edge_type not in self.VALID_EDGE_TYPES:
                    logger.warning(f"Invalid edge type in edge_weights: {edge_type}")
                if not 0.0 <= weight <= 1.0:
                    self.config.edge_weights[edge_type] = max(0.0, min(1.0, weight))

        # Validate neighbors_per_type
        if self.config.neighbors_per_type:
            for edge_type, count in self.config.neighbors_per_type.items():
                if edge_type not in self.VALID_EDGE_TYPES:
                    logger.warning(f"Invalid edge type in neighbors_per_type: {edge_type}")
                if not self.MIN_NEIGHBORS <= count <= self.MAX_NEIGHBORS:
                    self.config.neighbors_per_type[edge_type] = max(
                        self.MIN_NEIGHBORS, min(count, self.MAX_NEIGHBORS)
                    )

        # Validate numeric constraints
        if not (self.MIN_HOPS <= self.config.max_hops <= self.MAX_HOPS):
            self.config.max_hops = max(self.MIN_HOPS,
                                       min(self.config.max_hops, self.MAX_HOPS))

        if not (self.MIN_NEIGHBORS <= self.config.max_neighbors <= self.MAX_NEIGHBORS):
            self.config.max_neighbors = max(self.MIN_NEIGHBORS,
                                            min(self.config.max_neighbors, self.MAX_NEIGHBORS))

    def get_config(self) -> StrategyConfig:
        """Get the strategy configuration."""
        return self.config

    def update(self, **kwargs):
        """Update strategy parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary."""
        return self.config.to_dict()

    def to_json(self) -> str:
        """Convert strategy to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Strategy':
        """Create strategy from JSON string."""
        data = json.loads(json_str)
        config = StrategyConfig.from_dict(data)
        return cls(config)

    @classmethod
    def parse_from_llm_response(cls, response: str, fallback_strategy: Optional['Strategy'] = None) -> 'Strategy':
        """
        Parse strategy from LLM response with robust multi-stage parsing.

        Implements a robust parsing pipeline:
        1. Try JSON code block extraction
        2. Try raw JSON parsing
        3. Try key-value extraction from natural language
        4. Fall back to default/previous strategy

        Args:
            response: Raw LLM response text
            fallback_strategy: Strategy to use if parsing fails (preserves continuity)

        Returns:
            Parsed Strategy object
        """
        parsed_data = None
        parse_method = None

        # Stage 1: Try JSON code block patterns (most common LLM output format)
        json_patterns = [
            (r'```json\s*([\s\S]*?)\s*```', 'json_block'),      # ```json ... ```
            (r'```\s*([\s\S]*?)\s*```', 'code_block'),           # ``` ... ```
            (r'(\{[\s\S]*"edge_types"[\s\S]*\})', 'inline_json'), # Inline JSON with edge_types
        ]

        for pattern, method_name in json_patterns:
            match = re.search(pattern, response)
            if match:
                json_str = match.group(1).strip()
                try:
                    parsed_data = json.loads(json_str)
                    parse_method = method_name
                    logger.debug(f"Strategy parsed using {method_name}")
                    break
                except json.JSONDecodeError:
                    continue

        # Stage 2: Try parsing entire response as JSON
        if parsed_data is None:
            try:
                # Clean up common issues
                cleaned = response.strip()
                # Remove leading/trailing text before/after JSON
                json_match = re.search(r'\{[\s\S]*\}', cleaned)
                if json_match:
                    parsed_data = json.loads(json_match.group(0))
                    parse_method = 'full_json'
                    logger.debug("Strategy parsed as full JSON")
            except json.JSONDecodeError:
                pass

        # Stage 3: Key-value extraction from natural language
        if parsed_data is None:
            parsed_data = cls._extract_from_natural_language(response)
            if parsed_data:
                parse_method = 'natural_language'
                logger.debug("Strategy extracted from natural language")

        # Stage 4: Fall back to default or provided fallback
        if parsed_data is None:
            if fallback_strategy:
                logger.warning("Strategy parsing failed, using fallback strategy")
                return fallback_strategy
            else:
                logger.warning("Strategy parsing failed, using default strategy")
                return cls()

        # Validate and sanitize parsed data
        validated_data = cls._validate_and_sanitize(parsed_data)

        logger.info(f"Strategy parsed successfully via {parse_method}: "
                   f"edge_types={validated_data.get('edge_types', [])}")

        config = StrategyConfig.from_dict(validated_data)
        return cls(config)

    @classmethod
    def _extract_from_natural_language(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract strategy parameters from natural language response.

        Handles cases where LLM outputs prose instead of JSON.
        """
        data = {}

        # Extract edge_types
        edge_type_patterns = [
            r'edge[_\s]?types?[:\s]+\[([^\]]+)\]',
            r'edge[_\s]?types?[:\s]+([A-Z,\s]+)',
            r'(?:use|include|focus on|prioritize)\s+(?:the\s+)?([A-Z]+(?:\s*,\s*[A-Z]+)*)\s+(?:edges?|types?)',
        ]
        for pattern in edge_type_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                edge_str = match.group(1)
                # Parse edge types
                edges = re.findall(r'[A-Z][A-Za-z]+', edge_str)
                valid_edges = [e for e in edges if e in cls.VALID_EDGE_TYPES]
                if valid_edges:
                    data['edge_types'] = valid_edges
                    break

        # Extract max_hops
        hops_patterns = [
            r'max[_\s]?hops?[:\s]+(\d+)',
            r'(\d+)[_\s]?hops?',
            r'depth[:\s]+(\d+)',
        ]
        for pattern in hops_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['max_hops'] = int(match.group(1))
                break

        # Extract max_neighbors
        neighbors_patterns = [
            r'max[_\s]?neighbors?[:\s]+(\d+)',
            r'max[_\s]?neighbors?\s+(?:of\s+)?(\d+)',
            r'(\d+)\s+neighbors?',
            r'neighbors?[:\s]+(\d+)',
            r'neighbors?\s+(?:of|=|:)\s*(\d+)',
        ]
        for pattern in neighbors_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['max_neighbors'] = int(match.group(1))
                break

        # Extract sampling method
        sampling_patterns = [
            r'sampling[:\s]+["\']?(\w+)["\']?',
            r'(?:use|with)\s+(top_k|random|weighted)\s+sampling',
        ]
        for pattern in sampling_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                method = match.group(1).lower()
                if method in cls.VALID_SAMPLING:
                    data['sampling'] = method
                    break

        # Extract description_length
        desc_len_patterns = [
            r'description[_\s]?length[:\s]+["\']?(short|medium|long)["\']?',
            r'(?:use|prefer|set)\s+(short|medium|long)\s+description',
        ]
        for pattern in desc_len_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['description_length'] = match.group(1).lower()
                break

        # Extract include_statistics
        stats_patterns = [
            r'include[_\s]?statistics[:\s]+(true|false)',
            r'statistics[:\s]+(true|false|yes|no)',
        ]
        for pattern in stats_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).lower()
                data['include_statistics'] = val in ('true', 'yes')
                break

        # Extract focus_keywords
        keywords_patterns = [
            r'focus[_\s]?keywords?[:\s]+\[([^\]]+)\]',
            r'keywords?[:\s]+\[([^\]]+)\]',
        ]
        for pattern in keywords_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                kw_str = match.group(1)
                keywords = [kw.strip().strip('"\'') for kw in kw_str.split(',')]
                data['focus_keywords'] = [kw for kw in keywords if kw]
                break

        # Extract reasoning if present
        reasoning_patterns = [
            r'reasoning[:\s]+["\']?(.+?)["\']?(?:\n|$)',
            r'because[:\s]+(.+?)(?:\n|$)',
            r'rationale[:\s]+(.+?)(?:\n|$)',
        ]
        for pattern in reasoning_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['reasoning'] = match.group(1).strip()[:500]
                break

        return data if data else None

    @classmethod
    def _validate_and_sanitize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize parsed strategy data.
        """
        validated = {}

        # Validate edge_types
        if 'edge_types' in data:
            edge_types = data['edge_types']
            if isinstance(edge_types, list):
                validated['edge_types'] = [
                    et for et in edge_types if et in cls.VALID_EDGE_TYPES
                ]
            elif isinstance(edge_types, str):
                # Handle comma-separated string
                validated['edge_types'] = [
                    et.strip() for et in edge_types.split(',')
                    if et.strip() in cls.VALID_EDGE_TYPES
                ]

        if not validated.get('edge_types'):
            validated['edge_types'] = ['PPI', 'GO', 'HPO']

        # Validate max_hops
        if 'max_hops' in data:
            try:
                max_hops = int(data['max_hops'])
                validated['max_hops'] = max(cls.MIN_HOPS, min(max_hops, cls.MAX_HOPS))
            except (ValueError, TypeError):
                validated['max_hops'] = 2

        # Validate max_neighbors
        if 'max_neighbors' in data:
            try:
                max_neighbors = int(data['max_neighbors'])
                validated['max_neighbors'] = max(cls.MIN_NEIGHBORS, min(max_neighbors, cls.MAX_NEIGHBORS))
            except (ValueError, TypeError):
                validated['max_neighbors'] = 50

        # Validate sampling method
        if 'sampling' in data:
            sampling = str(data['sampling']).lower()
            validated['sampling'] = sampling if sampling in cls.VALID_SAMPLING else 'top_k'

        # Validate description_length
        if 'description_length' in data:
            desc_len = str(data['description_length']).lower()
            validated['description_length'] = desc_len if desc_len in cls.VALID_DESCRIPTION_LENGTHS else 'medium'

        # Validate edge_weights
        if 'edge_weights' in data and isinstance(data['edge_weights'], dict):
            validated['edge_weights'] = {}
            for edge_type, weight in data['edge_weights'].items():
                if edge_type in cls.VALID_EDGE_TYPES:
                    try:
                        w = float(weight)
                        validated['edge_weights'][edge_type] = max(0.0, min(1.0, w))
                    except (ValueError, TypeError):
                        pass

        # Validate neighbors_per_type
        if 'neighbors_per_type' in data and isinstance(data['neighbors_per_type'], dict):
            validated['neighbors_per_type'] = {}
            for edge_type, count in data['neighbors_per_type'].items():
                if edge_type in cls.VALID_EDGE_TYPES:
                    try:
                        c = int(count)
                        validated['neighbors_per_type'][edge_type] = max(cls.MIN_NEIGHBORS, min(c, cls.MAX_NEIGHBORS))
                    except (ValueError, TypeError):
                        pass

        # Validate include_statistics
        if 'include_statistics' in data:
            validated['include_statistics'] = bool(data['include_statistics'])

        # Validate focus_keywords
        if 'focus_keywords' in data:
            if isinstance(data['focus_keywords'], list):
                validated['focus_keywords'] = [str(kw).strip()[:50] for kw in data['focus_keywords'][:10]]
            elif isinstance(data['focus_keywords'], str):
                validated['focus_keywords'] = [kw.strip()[:50] for kw in data['focus_keywords'].split(',')][:10]

        # Preserve reasoning
        if 'reasoning' in data and isinstance(data['reasoning'], str):
            validated['reasoning'] = data['reasoning'][:500]

        return validated

    def __repr__(self) -> str:
        return f"Strategy({self.config})"

    @classmethod
    def parse_with_retry(
        cls,
        response: str,
        llm_client,
        fallback_strategy: Optional['Strategy'] = None,
        max_retries: int = 2
    ) -> 'Strategy':
        """
        Parse strategy from LLM response with retry on failure.

        If initial parsing fails or returns the fallback (indicating failure),
        requests the LLM to fix the format.

        Args:
            response: Raw LLM response text
            llm_client: LLM client for retry requests
            fallback_strategy: Strategy to use as fallback
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed Strategy object
        """
        for attempt in range(max_retries + 1):
            strategy = cls.parse_from_llm_response(response, fallback_strategy)

            # Check if parsing actually extracted new values
            # (not just returning the fallback unchanged)
            if strategy and fallback_strategy:
                strategy_dict = strategy.to_dict()
                fallback_dict = fallback_strategy.to_dict()
                # Compare key fields to see if we got new values
                key_fields = ['edge_types', 'max_hops', 'max_neighbors', 'sampling']
                has_changes = any(
                    strategy_dict.get(f) != fallback_dict.get(f)
                    for f in key_fields
                )
                if has_changes:
                    logger.info(f"Strategy parsed successfully on attempt {attempt + 1}")
                    return strategy
            elif strategy and not fallback_strategy:
                # No fallback to compare, assume success
                logger.info(f"Strategy parsed successfully on attempt {attempt + 1}")
                return strategy

            # Parsing failed or returned unchanged fallback, request LLM to fix
            if attempt < max_retries:
                logger.warning(f"Strategy parsing attempt {attempt + 1} failed, requesting LLM to fix format")

                repair_prompt = f"""Your previous response could not be parsed as a valid strategy JSON.

Previous response (truncated):
{response[:800]}

Please output a valid strategy JSON in exactly this format:
```json
{{
    "edge_types": ["PPI", "GO", "HPO"],
    "max_hops": 2,
    "sampling": "top_k",
    "max_neighbors": 50,
    "reasoning": "Your explanation here"
}}
```

Requirements:
- edge_types: Array of strings from [PPI, GO, HPO, TRRUST, CellMarker, Reactome]
- max_hops: Integer between 1-3
- sampling: One of "top_k", "random", "weighted"
- max_neighbors: Integer between 10-200
- reasoning: Brief explanation string

Output ONLY the JSON block, no additional text."""

                try:
                    if hasattr(llm_client, 'generate_strong'):
                        response = llm_client.generate_strong(repair_prompt)
                    else:
                        response = llm_client.generate(repair_prompt)
                except Exception as e:
                    logger.error(f"LLM repair request failed: {e}")
                    break

        # All retries exhausted, return fallback or default
        logger.warning(f"Strategy parsing failed after {max_retries + 1} attempts")
        return fallback_strategy or cls()


def get_default_strategy(task_name: str) -> Strategy:
    """
    Get task-specific default strategy with proven high-performance settings.

    Args:
        task_name: Name of the downstream task

    Returns:
        Strategy configured for the task with optimized parameters
    """
    from .prompts import TASK_EDGE_PRIORITIES, TASK_INITIAL_PROMPTS

    prompt_template = TASK_INITIAL_PROMPTS.get(task_name, "")

    # Task-specific optimized configurations based on empirical results
    if task_name == 'geneattribute_dosage_sensitivity':
        # Proven configuration achieving AUC 0.94+
        # Key insight: More context (max_neighbors=100, long desc) improves performance
        config = StrategyConfig(
            edge_types=['HPO', 'GO', 'PPI', 'TRRUST', 'Reactome'],
            max_hops=2,
            sampling='weighted',
            max_neighbors=100,  # Critical: more context = better performance
            description_length='long',  # Critical: longer descriptions capture more signals
            edge_weights={
                'HPO': 1.0,   # Phenotypes most informative for dosage sensitivity
                'GO': 0.8,    # Biological processes important
                'PPI': 0.6,   # Protein interactions for hub detection
                'TRRUST': 0.5,
                'Reactome': 0.5
            },
            neighbors_per_type={
                'HPO': 30,
                'GO': 25,
                'PPI': 20,
                'TRRUST': 15,
                'Reactome': 10
            },
            include_statistics=True,
            focus_keywords=[
                'haploinsufficiency', 'dosage sensitivity', 'pLI', 'LOEUF',
                'loss of function intolerance', 'gene essentiality', 'constraint',
                'hub gene', 'highly conserved', 'essential pathway'
            ],
            prompt_template=prompt_template,
            reasoning="Optimized configuration for dosage sensitivity based on empirical results"
        )
        return Strategy(config)

    # Default configuration for other tasks
    edge_types = TASK_EDGE_PRIORITIES.get(task_name, ['PPI', 'GO', 'HPO'])

    config = StrategyConfig(
        edge_types=edge_types,
        max_hops=2,
        sampling='top_k',
        max_neighbors=50,
        prompt_template=prompt_template,
        reasoning="Initial task-specific strategy"
    )

    return Strategy(config)


# =============================================================================
# Strategy Distance Calculation (策略距离计算)
# =============================================================================

# Distance weights for different parameters
DISTANCE_WEIGHTS = {
    'edge_types': 0.30,       # Most important - determines KG structure
    'max_neighbors': 0.20,    # Affects information density
    'max_hops': 0.15,         # Affects graph depth
    'sampling': 0.15,         # Affects selection strategy
    'description_length': 0.10,  # Affects output verbosity
    'other': 0.10,            # Combined smaller params
}

# Ordinal mappings for categorical parameters
DESCRIPTION_LENGTH_ORDINAL = {'short': 0, 'medium': 1, 'long': 2}
SAMPLING_ORDINAL = {'top_k': 0, 'random': 1, 'weighted': 2}


def compute_strategy_distance(
    strategy_curr: Dict[str, Any],
    strategy_prev: Optional[Dict[str, Any]]
) -> float:
    """
    计算两个策略之间的归一化距离 (Normalized distance between strategies)

    用于科学发现范式中判断"勇敢探索"还是"懒惰重复"

    计算组件:
    - edge_types: Jaccard 距离 (集合差异)
    - max_hops: |curr - prev| / 2 (按范围归一化)
    - max_neighbors: |curr - prev| / 190 (范围 10-200)
    - sampling: 1 如果不同, 0 如果相同
    - description_length: 序数距离 / 2

    Args:
        strategy_curr: 当前策略字典
        strategy_prev: 上一个策略字典（如果为 None，返回 1.0 表示完全新策略）

    Returns:
        float: 距离值 [0, 1]，其中 0 = 完全相同，1 = 最大差异
    """
    if strategy_prev is None:
        return 1.0  # 第一次迭代，完全"新"策略

    total_distance = 0.0

    # 1. Edge types distance (Jaccard distance)
    curr_edges = set(strategy_curr.get('edge_types', []))
    prev_edges = set(strategy_prev.get('edge_types', []))

    if curr_edges or prev_edges:
        intersection = len(curr_edges & prev_edges)
        union = len(curr_edges | prev_edges)
        jaccard_distance = 1 - (intersection / union) if union > 0 else 0
        total_distance += DISTANCE_WEIGHTS['edge_types'] * jaccard_distance

    # 2. Max neighbors distance (normalized by range 10-200)
    curr_neighbors = strategy_curr.get('max_neighbors', 50)
    prev_neighbors = strategy_prev.get('max_neighbors', 50)
    neighbors_distance = abs(curr_neighbors - prev_neighbors) / 190.0  # Range is 190 (10-200)
    total_distance += DISTANCE_WEIGHTS['max_neighbors'] * min(neighbors_distance, 1.0)

    # 3. Max hops distance (normalized by range 1-3)
    curr_hops = strategy_curr.get('max_hops', 2)
    prev_hops = strategy_prev.get('max_hops', 2)
    hops_distance = abs(curr_hops - prev_hops) / 2.0  # Range is 2 (1-3)
    total_distance += DISTANCE_WEIGHTS['max_hops'] * hops_distance

    # 4. Sampling method distance (categorical)
    curr_sampling = strategy_curr.get('sampling', 'top_k')
    prev_sampling = strategy_prev.get('sampling', 'top_k')
    if curr_sampling != prev_sampling:
        # Ordinal distance for sampling
        curr_ord = SAMPLING_ORDINAL.get(curr_sampling, 0)
        prev_ord = SAMPLING_ORDINAL.get(prev_sampling, 0)
        sampling_distance = abs(curr_ord - prev_ord) / 2.0
    else:
        sampling_distance = 0.0
    total_distance += DISTANCE_WEIGHTS['sampling'] * sampling_distance

    # 5. Description length distance (ordinal)
    curr_length = strategy_curr.get('description_length', 'medium')
    prev_length = strategy_prev.get('description_length', 'medium')
    curr_ord = DESCRIPTION_LENGTH_ORDINAL.get(curr_length, 1)
    prev_ord = DESCRIPTION_LENGTH_ORDINAL.get(prev_length, 1)
    length_distance = abs(curr_ord - prev_ord) / 2.0  # Range is 2 (short to long)
    total_distance += DISTANCE_WEIGHTS['description_length'] * length_distance

    # 6. Other parameters (combined)
    other_distance = 0.0
    other_params = [
        ('description_focus', 'balanced'),
        ('context_window', 'full'),
        ('prompt_style', 'analytical'),
        ('feature_selection', 'all'),
    ]
    changed_count = sum(
        1 for param, default in other_params
        if strategy_curr.get(param, default) != strategy_prev.get(param, default)
    )
    other_distance = changed_count / len(other_params)
    total_distance += DISTANCE_WEIGHTS['other'] * other_distance

    # Ensure result is in [0, 1]
    return min(max(total_distance, 0.0), 1.0)
