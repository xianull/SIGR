"""
Task-Specific Prompts and Edge Priorities for SIGR Actor

Defines initial prompts and edge type priorities for each downstream task.
All prompts are in English with anti-leakage constraints.

Key design principles:
1. CONCISE descriptions (100-150 words) - avoid token waste
2. FOCUSED on downstream task-relevant features
3. IMPLICIT signals allowed (hub gene, conserved, etc.)
4. EXPLICIT labels forbidden (is/is not sensitive, etc.)
"""

from typing import Dict, List, Any, Optional

__all__ = [
    # Constants
    'TASK_EDGE_PRIORITIES',
    'TASK_INITIAL_PROMPTS',
    'TASK_IMPLICIT_SIGNALS',
    'MODE_CONSTRAINTS',
    'TASK_DIAGNOSIS_QUESTIONS',
    # Functions
    'get_reflection_prompt',
    'get_prompt_optimization_prompt',
    'get_self_critique_prompt',
    'get_consistency_check_prompt',
    'get_reflection_cot_prompt',
    'get_scientist_reflection_prompt',
    'get_biologist_reflection_prompt',
    'get_bio_cot_prompt',
    'format_strategy_summary',
]


# Task-specific edge type priorities
# These define which edge types are most relevant for each task
TASK_EDGE_PRIORITIES: Dict[str, List[str]] = {
    'ppi': ['PPI', 'GO', 'Reactome', 'TRRUST'],  # Protein interactions
    'genetype': ['GO', 'Reactome', 'PPI', 'HPO'],  # Gene Ontology is primary
    'phenotype': ['HPO', 'GO', 'Reactome', 'PPI'],  # Phenotype associations
    'celltype': ['CellMarker', 'GO', 'Reactome', 'PPI'],  # Cell type markers
    'dosage': ['TRRUST', 'GO', 'Reactome', 'PPI'],  # Regulatory
    # GGI task
    'ggi': ['PPI', 'GO', 'Reactome', 'TRRUST'],  # Gene-gene interaction
    # Cell task
    'cell': ['CellMarker', 'GO', 'Reactome', 'PPI'],  # Cell type
    # GeneAttribute subtasks
    'geneattribute_dosage_sensitivity': ['TRRUST', 'GO', 'HPO', 'Reactome', 'PPI'],  # Essentiality
    'geneattribute_lys4_only': ['GO', 'Reactome', 'TRRUST', 'HPO'],  # Chromatin/methylation
    'geneattribute_no_methylation': ['GO', 'Reactome', 'TRRUST', 'HPO'],  # Chromatin/methylation
    'geneattribute_bivalent': ['GO', 'Reactome', 'TRRUST', 'HPO'],  # Chromatin state
}


# Task-specific IMPLICIT signals that should be emphasized
# These are biological features that IMPLICITLY indicate the label without stating it directly
TASK_IMPLICIT_SIGNALS: Dict[str, List[str]] = {
    'ppi': [
        'interaction partners', 'binding domain', 'protein complex',
        'signaling hub', 'scaffold protein', 'co-localization',
        'shared pathway', 'functional module'
    ],
    'genetype': [
        'enzymatic activity', 'catalytic domain', 'binding capacity',
        'transcription factor', 'kinase', 'receptor',
        'transporter', 'channel protein'
    ],
    'phenotype': [
        'disease association', 'clinical manifestation', 'mutation effect',
        'loss-of-function', 'gain-of-function', 'knockout phenotype',
        'developmental defect', 'syndrome'
    ],
    'celltype': [
        'tissue-specific', 'cell-type marker', 'enriched in',
        'lineage marker', 'differentiation', 'cluster marker',
        'single-cell expression', 'cell identity'
    ],
    'dosage': [
        'hub gene', 'highly conserved', 'essential pathway',
        'haploinsufficiency', 'regulatory target', 'constraint',
        'core complex', 'network centrality'
    ],
    'ggi': [
        'co-expression', 'shared pathway', 'functional module',
        'interaction network', 'regulatory cascade', 'complex member',
        'epistatic interaction', 'synthetic lethal'
    ],
    'cell': [
        'tissue-specific', 'marker gene', 'enriched expression',
        'cell identity', 'differentiation marker', 'lineage-specific',
        'cluster-defining', 'canonical marker'
    ],
    'geneattribute_dosage_sensitivity': [
        'hub gene', 'highly connected', 'central node',
        'highly conserved', 'evolutionary constrained', 'pLI',
        'essential pathway', 'core complex member', 'LOEUF',
        'regulatory target', 'transcription factor target',
        'haploinsufficiency', 'dosage-dependent'
    ],
    'geneattribute_lys4_only': [
        'H3K4me3', 'active promoter', 'housekeeping gene',
        'constitutive expression', 'open chromatin',
        'transcriptionally active', 'CpG island'
    ],
    'geneattribute_no_methylation': [
        'unmethylated', 'CpG island', 'promoter accessibility',
        'active transcription', 'housekeeping', 'ubiquitous expression',
        'open chromatin state'
    ],
    'geneattribute_bivalent': [
        'H3K4me3', 'H3K27me3', 'poised promoter', 'bivalent domain',
        'developmental gene', 'Polycomb target', 'PRC2',
        'lineage commitment', 'pluripotency', 'differentiation potential',
        'epigenetic plasticity'
    ],
}


# Task-specific optimization hints based on empirical results
# These hints guide the LLM to use proven high-performance configurations
TASK_OPTIMIZATION_HINTS: Dict[str, str] = {
    'geneattribute_dosage_sensitivity': """
CRITICAL OPTIMIZATION HINTS for dosage sensitivity (from empirical results):
- max_neighbors=100 has achieved AUC 0.94+ - DO NOT reduce below 80
- description_length='long' captures more constraint signals - PREFER 'long'
- HPO edges are MOST informative - prioritize HPO with weight 1.0
- weighted sampling outperforms top_k for this task
- Key focus_keywords: haploinsufficiency, pLI, LOEUF, constraint, hub gene
- Target AUC: 0.95+ (baseline GenePT: 0.93)
""",
    'geneattribute_lys4_only': """
OPTIMIZATION HINTS for lys4_only task:
- Focus on GO and Reactome edges for chromatin/methylation information
- description_length='medium' is usually sufficient
- Key focus_keywords: H3K4me3, active promoter, housekeeping
""",
    'geneattribute_bivalent': """
OPTIMIZATION HINTS for bivalent task:
- Prioritize GO edges for developmental and epigenetic information
- Key focus_keywords: H3K4me3, H3K27me3, poised promoter, PRC2, pluripotency
""",
}


# Task-specific initial prompts - OPTIMIZED for conciseness and task-relevance
# Key changes: 100-150 word limit, focused on task-specific implicit signals
TASK_INITIAL_PROMPTS: Dict[str, str] = {
    'ppi': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for protein interaction prediction.

KG Context:
- Interactions: {ppi_info}
- Pathways: {pathway_info}
- Processes: {go_info}

FOCUS on interaction-relevant features:
- Protein domains, binding sites
- Known interaction partners count
- Complex membership
- Signaling pathway hub status

Include IMPLICIT signals: "hub protein", "binding domain", "scaffold", "complex member"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit predictions ("will/won't interact")
- NO probability statements
- Factual biological features only
""",

    'genetype': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for gene functional type classification.

KG Context:
- Functions: {go_info}
- Pathways: {pathway_info}

FOCUS on functional type indicators:
- Molecular function (enzyme, receptor, transporter, etc.)
- Catalytic/binding domains
- Subcellular localization
- Pathway roles

Include IMPLICIT signals: "kinase activity", "receptor function", "transcription factor", "channel protein"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit type predictions
- NO classification labels
- Factual functional features only
""",

    'phenotype': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for phenotype association prediction.

KG Context:
- Phenotypes: {phenotype_info}
- Processes: {go_info}

FOCUS on phenotype-relevant features:
- Known disease associations
- Mutation effects in model organisms
- Clinical manifestations
- Developmental roles

Include IMPLICIT signals: "disease-associated", "knockout phenotype", "syndrome", "developmental defect"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit phenotype predictions
- NO probability statements
- Factual phenotypic features only
""",

    'celltype': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for cell type marker prediction.

KG Context:
- Cell markers: {celltype_info}
- Expression: {tissue_info}
- Processes: {go_info}

FOCUS on cell-type relevant features:
- Tissue-specific expression patterns
- Enrichment in specific cell populations
- Differentiation markers
- Lineage associations

Include IMPLICIT signals: "enriched in", "tissue-specific", "marker for", "lineage-defining"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit marker predictions
- NO cell type labels
- Factual expression features only
""",

    'dosage': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for dosage sensitivity prediction.

KG Context:
- Regulatory: {tf_info}
- Processes: {go_info}
- Interactions: {ppi_info}

FOCUS on dosage-relevant features:
- Network centrality (hub status)
- Conservation level
- Essential pathway membership
- Regulatory network position

Include IMPLICIT signals: "hub gene", "highly conserved", "essential", "haploinsufficiency context"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit sensitivity predictions
- NO dosage labels
- Factual essentiality features only
""",

    'ggi': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for gene-gene interaction prediction.

KG Context:
- Interactions: {ppi_info}
- Pathways: {pathway_info}
- Regulatory: {tf_info}

FOCUS on interaction-relevant features:
- Known interaction network position
- Co-expression patterns
- Shared pathway membership
- Regulatory relationships

Include IMPLICIT signals: "co-expressed with", "shared pathway", "functional module", "regulatory cascade"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit interaction predictions
- NO classification labels
- Factual interaction features only
""",

    'cell': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for cell type classification.

KG Context:
- Cell markers: {celltype_info}
- Processes: {go_info}
- Interactions: {ppi_info}

FOCUS on cell-type relevant features:
- Expression specificity
- Marker gene characteristics
- Tissue distribution
- Cell identity associations

Include IMPLICIT signals: "cell-type specific", "marker gene", "enriched expression", "cluster-defining"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit cell type predictions
- NO classification labels
- Factual expression features only
""",

    'geneattribute_dosage_sensitivity': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for dosage sensitivity prediction.

KG Context:
- Regulatory network: {tf_info}
- Biological processes: {go_info}
- Phenotypes: {phenotype_info}
- Pathways: {pathway_info}
- Interactions: {ppi_info}

FOCUS on dosage sensitivity indicators:
- Network centrality (hub gene status, interaction count)
- Evolutionary conservation (constraint scores context)
- Essential pathway membership
- Regulatory target complexity
- Haploinsufficiency-related features

Include IMPLICIT signals: "hub gene", "highly conserved", "essential pathway", "core complex", "regulatory target"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit predictions ("is/is not dosage-sensitive")
- NO probability or confidence statements
- Focus on RELEVANT features that indicate essentiality
""",

    'geneattribute_lys4_only': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for H3K4me3-only chromatin state prediction.

KG Context:
- Processes: {go_info}
- Regulatory: {tf_info}
- Phenotypes: {phenotype_info}

FOCUS on active chromatin indicators:
- Transcriptional activity level
- Housekeeping gene characteristics
- Promoter features
- Expression breadth

Include IMPLICIT signals: "constitutively expressed", "active promoter", "housekeeping", "ubiquitous expression"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit chromatin state predictions
- NO methylation labels
- Factual transcriptional features only
""",

    'geneattribute_no_methylation': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for unmethylated chromatin state prediction.

KG Context:
- Processes: {go_info}
- Regulatory: {tf_info}
- Phenotypes: {phenotype_info}

FOCUS on unmethylated promoter indicators:
- CpG island association
- Promoter accessibility
- Constitutive expression patterns
- Open chromatin features

Include IMPLICIT signals: "CpG island", "open chromatin", "active transcription", "accessible promoter"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit methylation predictions
- NO chromatin state labels
- Factual promoter features only
""",

    'geneattribute_bivalent': """Generate a CONCISE description (100-150 words) for gene {gene_id} ({gene_name}) for bivalent chromatin state prediction.

KG Context:
- Processes: {go_info}
- Regulatory: {tf_info}
- Phenotypes: {phenotype_info}

FOCUS on bivalent domain indicators:
- Developmental gene characteristics
- Lineage commitment roles
- Pluripotency associations
- Polycomb regulation context

Include IMPLICIT signals: "developmental gene", "lineage commitment", "Polycomb target", "differentiation potential", "poised"

STRICT CONSTRAINTS:
- Maximum 150 words
- NO explicit bivalent predictions
- NO chromatin state labels ("is/is not bivalent")
- Factual developmental features only
""",
}


# Template for the Actor's reflection prompt
REFLECTION_PROMPT_TEMPLATE = """You are optimizing a gene representation strategy for the {task_name} task.

Current Strategy:
- Edge types: {edge_types}
- Max hops: {max_hops}
- Sampling method: {sampling}
- Max neighbors: {max_neighbors}
- Description length: {description_length}
- Edge weights: {edge_weights}
- Neighbors per type: {neighbors_per_type}
- Include statistics: {include_statistics}
- Focus keywords: {focus_keywords}

Current Reward: {reward:.4f}

{trend_analysis_section}

Feedback from Evaluator:
{feedback}

History (last {history_count} iterations):
{history}

Based on the feedback and trend analysis, analyze:
1. What aspects of the current strategy are working well?
2. What aspects need improvement?
3. How should the strategy be modified to improve performance?
4. Given the trend, should we explore new directions or fine-tune the current approach?

Available parameters:
- Edge types: PPI, GO, HPO, TRRUST, CellMarker, Reactome
- Sampling: top_k, random, weighted
- Max hops: 1-3
- Max neighbors: 10-200
- Description length: short (50-100 words), medium (100-150 words), long (150-250 words)
- Edge weights: 0.0-1.0 per edge type (higher = more important in formatting)
- Neighbors per type: 10-200 per edge type (fine-grained control)
- Include statistics: true/false (whether to include counts, scores in description)
- Focus keywords: list of terms to emphasize (e.g., "hub", "conserved", "essential")

Edge type descriptions:
- PPI: Protein-protein interactions (STRING)
- GO: Gene Ontology biological processes
- HPO: Human Phenotype Ontology associations
- TRRUST: Transcription factor regulatory relations
- CellMarker: Cell type marker annotations
- Reactome: Biological pathway membership

Output a JSON with the updated strategy:
{{
    "edge_types": [...],
    "max_hops": int,
    "sampling": "top_k" | "random" | "weighted",
    "max_neighbors": int,
    "description_length": "short" | "medium" | "long",
    "edge_weights": {{"PPI": 1.0, "GO": 0.8, ...}},
    "neighbors_per_type": {{"PPI": 30, "GO": 20, ...}},
    "include_statistics": true | false,
    "focus_keywords": ["keyword1", "keyword2", ...],
    "reasoning": "explanation of changes"
}}
"""


# Template for prompt optimization
PROMPT_OPTIMIZATION_TEMPLATE = """The current prompt template for the {task_name} task is not performing well.

Current prompt:
{current_prompt}

Recent feedback:
{feedback}

Current reward: {reward:.4f}

Generate an improved prompt template that:
1. Focuses more on information relevant to {task_name}
2. Maintains the anti-leakage constraints (no predictions, no labels, factual only)
3. Produces more informative gene descriptions
4. Uses the placeholders: {{gene_id}}, {{gene_name}}, {{ppi_info}}, {{go_info}}, {{phenotype_info}}, {{tf_info}}, {{celltype_info}}

Output only the new prompt template, no additional explanation.
"""


def get_reflection_prompt(
    task_name: str,
    strategy_dict: dict,
    reward: float,
    feedback: str,
    history: list,
    max_history: int = 3,
    trend_analysis: dict = None
) -> str:
    """
    Generate the reflection prompt for the Actor.

    Args:
        task_name: Name of the downstream task
        strategy_dict: Current strategy as dictionary
        reward: Current reward value
        feedback: Feedback from evaluator
        history: List of (strategy, reward, feedback) tuples
        max_history: Maximum number of history items to include
        trend_analysis: Optional trend analysis from TrendAnalyzer

    Returns:
        Formatted reflection prompt
    """
    # Format history
    recent_history = history[-max_history:] if history else []
    history_str = ""
    for i, item in enumerate(recent_history):
        history_str += f"\nIteration {i+1}:\n"
        history_str += f"  - Edge types: {item['strategy'].get('edge_types', [])}\n"
        history_str += f"  - Reward: {item['reward']:.4f}\n"
        history_str += f"  - Key feedback: {item['feedback'][:200]}...\n"

    if not history_str:
        history_str = "No previous iterations."

    # Format trend analysis section
    trend_section = ""
    if trend_analysis:
        trend_section = f"""
Trend Analysis:
- Direction: {trend_analysis.get('trend_direction', 'unknown')} (strength: {trend_analysis.get('trend_strength', 0):.2f})
- Convergence: {trend_analysis.get('convergence_score', 0):.2f}
- Duration: {trend_analysis.get('trend_duration', 0)} iterations
- Suggested Action: {trend_analysis.get('suggested_action', 'explore')}
- Reason: {trend_analysis.get('action_reason', 'N/A')}
"""
        # Add effective strategies if available
        effective = trend_analysis.get('effective_strategies', [])
        if effective:
            trend_section += "\nHistorically effective patterns:\n"
            for e in effective[:3]:
                trend_section += f"  - {e.get('strategy', {})} (improvement: {e.get('improvement', 0):.4f})\n"

    # Format extended parameters
    edge_weights = strategy_dict.get('edge_weights', {})
    neighbors_per_type = strategy_dict.get('neighbors_per_type', {})
    focus_keywords = strategy_dict.get('focus_keywords', [])

    return REFLECTION_PROMPT_TEMPLATE.format(
        task_name=task_name,
        edge_types=strategy_dict.get('edge_types', []),
        max_hops=strategy_dict.get('max_hops', 2),
        sampling=strategy_dict.get('sampling', 'top_k'),
        max_neighbors=strategy_dict.get('max_neighbors', 50),
        description_length=strategy_dict.get('description_length', 'medium'),
        edge_weights=edge_weights if edge_weights else 'Not set (using defaults)',
        neighbors_per_type=neighbors_per_type if neighbors_per_type else 'Not set (using max_neighbors)',
        include_statistics=strategy_dict.get('include_statistics', True),
        focus_keywords=focus_keywords if focus_keywords else 'Not set',
        reward=reward,
        trend_analysis_section=trend_section,
        feedback=feedback,
        history_count=len(recent_history),
        history=history_str
    )


def get_prompt_optimization_prompt(
    task_name: str,
    current_prompt: str,
    feedback: str,
    reward: float
) -> str:
    """
    Generate the prompt optimization prompt.

    Args:
        task_name: Name of the downstream task
        current_prompt: Current prompt template
        feedback: Feedback from evaluator
        reward: Current reward value

    Returns:
        Formatted prompt optimization prompt
    """
    return PROMPT_OPTIMIZATION_TEMPLATE.format(
        task_name=task_name,
        current_prompt=current_prompt,
        feedback=feedback,
        reward=reward
    )


# ============================================================================
# Self-Critique and Chain-of-Thought Prompts
# ============================================================================

# Self-critique prompt for validating Actor's reasoning
SELF_CRITIQUE_PROMPT_TEMPLATE = """Review your analysis and proposed strategy:

{initial_analysis}

Self-Verification Checklist:
1. ROOT CAUSE ANALYSIS: Did you identify the actual root cause of the performance issue, or just symptoms?
2. HISTORICAL LEARNING: Did you consider which past changes were effective vs ineffective?
3. CHANGE MAGNITUDE: Is the proposed change too aggressive (>50% parameter change) or too conservative (<5%)?
4. CONSISTENCY: Does this change undo any previously successful modifications?
5. LOGICAL SOUNDNESS: Is the reasoning chain logically valid?

Rate your confidence (1-10) for this analysis.

If confidence < 7 or any checklist item fails:
- Explain what was wrong
- Provide a revised analysis
- Output an UPDATED strategy JSON

If confidence >= 7 and all items pass:
- Confirm the analysis is sound
- Output the SAME strategy JSON

Output format:
```json
{{
    "confidence": <1-10>,
    "issues_found": ["list of issues if any"],
    "edge_types": [...],
    "max_hops": int,
    "sampling": "top_k" | "random" | "weighted",
    "max_neighbors": int,
    "description_length": "short" | "medium" | "long",
    "edge_weights": {{}},
    "neighbors_per_type": {{}},
    "include_statistics": true | false,
    "focus_keywords": [...],
    "reasoning": "final reasoning after self-critique"
}}
```
"""


# Consistency check prompt
CONSISTENCY_CHECK_PROMPT_TEMPLATE = """Compare the proposed strategy change against historical best performance.

Historical Best Strategy (reward: {best_reward:.4f}):
{best_strategy}

Proposed New Strategy:
{proposed_strategy}

Questions to answer:
1. Which parameters have changed?
2. Are any changes reverting successful modifications from the best strategy?
3. Is the magnitude of change reasonable (recommended: <30% per parameter)?
4. Could these changes cause oscillation (e.g., repeatedly toggling the same parameter)?

If issues are detected:
- Suggest a more conservative modification that preserves successful elements
- Output a revised strategy JSON

If no issues:
- Confirm the change is safe
- Output the proposed strategy JSON unchanged

Output format:
```json
{{
    "changes_detected": ["list of changed parameters"],
    "potential_issues": ["list of concerns if any"],
    "recommendation": "proceed" | "revise",
    "edge_types": [...],
    "max_hops": int,
    "sampling": "top_k" | "random" | "weighted",
    "max_neighbors": int,
    "description_length": "short" | "medium" | "long",
    "edge_weights": {{}},
    "neighbors_per_type": {{}},
    "include_statistics": true | false,
    "focus_keywords": [...],
    "reasoning": "explanation"
}}
```
"""


# Chain-of-Thought enhanced reflection prompt
REFLECTION_COT_PROMPT_TEMPLATE = """You are optimizing a gene representation strategy for the {task_name} task.
Use chain-of-thought reasoning to analyze the situation step by step.

{optimization_hints}

## CRITICAL: NO IMPROVEMENT = PENALTY
**IMPORTANT**: The reward system penalizes stagnation. If you do not improve upon the current best metric:
- Staying at the same performance = NEGATIVE REWARD (punishment)
- The longer you stay without improvement, the HIGHER the penalty
- You MUST try different configurations to escape local optima
- Small conservative changes are likely to be punished!

## STEP 1: DIAGNOSE THE PROBLEM
Current Reward: {reward:.4f}

Feedback from Evaluator:
{feedback}

What is the ROOT CAUSE of the current performance? (List 1-2 core issues, not symptoms)

## STEP 2: HISTORICAL ANALYSIS
Recent iterations:
{history}

Which past changes were EFFECTIVE (improved performance)?
Which past changes were INEFFECTIVE (hurt performance or had no effect)?

## STEP 3: TREND ASSESSMENT
{trend_analysis_section}

Given the trend, should we:
- EXPLORE: Try significantly different parameters (plateau/declining trend) - RECOMMENDED if no recent improvement
- FINE-TUNE: Make small adjustments to current approach (only if actively improving)
- PRESERVE: Keep successful elements, only fix weaknesses
{edge_effects_section}
## STEP 4: IMPROVEMENT PLAN
Current Strategy:
- Edge types: {edge_types}
- Max hops: {max_hops}
- Sampling: {sampling}
- Max neighbors: {max_neighbors}
- Description length: {description_length}
- Edge weights: {edge_weights}
- Neighbors per type: {neighbors_per_type}
- Include statistics: {include_statistics}
- Focus keywords: {focus_keywords}
- Description focus: {description_focus}
- Context window: {context_window}
- Prompt style: {prompt_style}
- Feature selection: {feature_selection}
- Generation passes: {generation_passes}

Based on Steps 1-3, propose improvements:
- What specific parameter(s) to change?
- Why this change addresses the root cause?
- Expected effect of the change?
- **If the trend is plateau or declining, you MUST make significant changes (>20% parameter modification)**

## STEP 5: OUTPUT STRATEGY
Available parameters:
- Edge types: PPI, GO, HPO, TRRUST, CellMarker, Reactome
- Sampling: top_k, random, weighted
- Max hops: 1-3
- Max neighbors: 10-200
- Description length: short (50-100 words), medium (100-150 words), long (150-250 words)
- Edge weights: 0.0-1.0 per edge type
- Neighbors per type: 10-200 per edge type
- Include statistics: true/false
- Focus keywords: list of terms to emphasize
- Description focus: balanced (default), network (emphasize hub/centrality), function (emphasize pathways), phenotype (emphasize disease/phenotype)
- Context window: minimal (direct neighbors only), local (1-hop), extended (2-hop), full (all available)
- Prompt style: analytical (factual analysis), narrative (story-like), structured (bullet points), comparative (compare with similar genes)
- Feature selection: all (include everything), essential (core features only), diverse (maximize variety), task_specific (focus on task-relevant features)
- Generation passes: 1-3 (number of refinement iterations)

```json
{{
    "edge_types": [...],
    "max_hops": int,
    "sampling": "top_k" | "random" | "weighted",
    "max_neighbors": int,
    "description_length": "short" | "medium" | "long",
    "edge_weights": {{"PPI": 1.0, ...}},
    "neighbors_per_type": {{"PPI": 30, ...}},
    "include_statistics": true | false,
    "focus_keywords": [...],
    "description_focus": "balanced" | "network" | "function" | "phenotype",
    "context_window": "minimal" | "local" | "extended" | "full",
    "prompt_style": "analytical" | "narrative" | "structured" | "comparative",
    "feature_selection": "all" | "essential" | "diverse" | "task_specific",
    "generation_passes": 1 | 2 | 3,
    "reasoning": "Summary of your Step 1-4 analysis"
}}
```
"""


def get_self_critique_prompt(initial_analysis: str) -> str:
    """
    Generate the self-critique prompt for validating Actor's reasoning.

    Args:
        initial_analysis: The initial reflection/analysis from the LLM

    Returns:
        Formatted self-critique prompt
    """
    return SELF_CRITIQUE_PROMPT_TEMPLATE.format(
        initial_analysis=initial_analysis
    )


def get_consistency_check_prompt(
    proposed_strategy: dict,
    best_strategy: dict,
    best_reward: float
) -> str:
    """
    Generate the consistency check prompt.

    Args:
        proposed_strategy: The proposed new strategy
        best_strategy: The historical best strategy
        best_reward: The reward achieved by best strategy

    Returns:
        Formatted consistency check prompt
    """
    import json
    return CONSISTENCY_CHECK_PROMPT_TEMPLATE.format(
        proposed_strategy=json.dumps(proposed_strategy, indent=2),
        best_strategy=json.dumps(best_strategy, indent=2),
        best_reward=best_reward
    )


def get_reflection_cot_prompt(
    task_name: str,
    strategy_dict: dict,
    reward: float,
    feedback: str,
    history: list,
    max_history: int = 3,
    trend_analysis: dict = None,
    best_reward: float = None,
    edge_effects: dict = None
) -> str:
    """
    Generate the Chain-of-Thought enhanced reflection prompt.

    Args:
        task_name: Name of the downstream task
        strategy_dict: Current strategy as dictionary
        reward: Current reward value
        feedback: Feedback from evaluator
        history: List of (strategy, reward, feedback) tuples
        max_history: Maximum number of history items to include
        trend_analysis: Optional trend analysis from TrendAnalyzer
        best_reward: Optional best reward for filtering effective history
        edge_effects: Optional edge type effectiveness from Memory

    Returns:
        Formatted CoT reflection prompt
    """
    # Filter history to prioritize effective changes
    # Only show iterations that achieved close to best performance (>= 99% of best)
    if history and best_reward is not None and best_reward > 0:
        effective_history = [h for h in history if h['reward'] >= best_reward * 0.99]
        if not effective_history:
            # If all are ineffective, show only the most recent 2 entries
            effective_history = history[-2:] if len(history) >= 2 else history
    else:
        effective_history = history

    # Apply max_history limit
    recent_history = effective_history[-max_history:] if effective_history else []
    history_str = ""

    for i, item in enumerate(recent_history):
        prev_reward = recent_history[i-1]['reward'] if i > 0 else None
        curr_reward = item['reward']

        if prev_reward is not None:
            if curr_reward > prev_reward:
                effect = "EFFECTIVE (+{:.4f})".format(curr_reward - prev_reward)
            elif curr_reward < prev_reward:
                effect = "INEFFECTIVE ({:.4f})".format(curr_reward - prev_reward)
            else:
                effect = "NEUTRAL (0.0)"
        else:
            effect = "BASELINE"

        history_str += f"\nIteration {i+1} [{effect}]:\n"
        history_str += f"  - Edge types: {item['strategy'].get('edge_types', [])}\n"
        history_str += f"  - Max neighbors: {item['strategy'].get('max_neighbors', 50)}\n"
        history_str += f"  - Description length: {item['strategy'].get('description_length', 'medium')}\n"
        history_str += f"  - Reward: {item['reward']:.4f}\n"
        history_str += f"  - Key feedback: {item['feedback'][:150]}...\n"

    if not history_str:
        history_str = "No previous iterations (this is the first iteration)."

    # Format trend analysis section
    trend_section = ""
    if trend_analysis:
        trend_section = f"""
Current Trend: {trend_analysis.get('trend_direction', 'unknown')} (strength: {trend_analysis.get('trend_strength', 0):.2f})
Convergence Score: {trend_analysis.get('convergence_score', 0):.2f}
Suggested Action: {trend_analysis.get('suggested_action', 'explore')}
Reason: {trend_analysis.get('action_reason', 'N/A')}
"""
    else:
        trend_section = "No trend data available (insufficient history)."

    # Format extended parameters
    edge_weights = strategy_dict.get('edge_weights', {})
    neighbors_per_type = strategy_dict.get('neighbors_per_type', {})
    focus_keywords = strategy_dict.get('focus_keywords', [])

    # Get task-specific optimization hints
    optimization_hints = TASK_OPTIMIZATION_HINTS.get(task_name, "")
    if optimization_hints:
        optimization_hints = f"## IMPORTANT - TASK-SPECIFIC OPTIMIZATION HINTS\n{optimization_hints}"

    # Format edge effects section
    edge_effects_section = ""
    if edge_effects:
        sorted_effects = sorted(
            edge_effects.items(),
            key=lambda x: x[1].get('ema_effect', 0),
            reverse=True
        )
        effects_lines = []
        for edge_type, effect in sorted_effects:
            usage = effect.get('usage_count', 0)
            success = effect.get('success_count', 0)
            success_rate = success / max(usage, 1)
            ema = effect.get('ema_effect', 0)
            effects_lines.append(
                f"  - {edge_type}: usage={usage}, success_rate={success_rate:.0%}, ema_effect={ema:+.4f}"
            )
        edge_effects_section = "\n## LEARNED EDGE EFFECTIVENESS (from Memory)\n" + "\n".join(effects_lines)
        edge_effects_section += "\n(Higher ema_effect = better historical performance. Consider prioritizing high-ema edge types.)\n"

    return REFLECTION_COT_PROMPT_TEMPLATE.format(
        task_name=task_name,
        optimization_hints=optimization_hints,
        edge_types=strategy_dict.get('edge_types', []),
        max_hops=strategy_dict.get('max_hops', 2),
        sampling=strategy_dict.get('sampling', 'top_k'),
        max_neighbors=strategy_dict.get('max_neighbors', 50),
        description_length=strategy_dict.get('description_length', 'medium'),
        edge_weights=edge_weights if edge_weights else 'Not set',
        neighbors_per_type=neighbors_per_type if neighbors_per_type else 'Not set',
        include_statistics=strategy_dict.get('include_statistics', True),
        focus_keywords=focus_keywords if focus_keywords else 'Not set',
        description_focus=strategy_dict.get('description_focus', 'balanced'),
        context_window=strategy_dict.get('context_window', 'full'),
        prompt_style=strategy_dict.get('prompt_style', 'analytical'),
        feature_selection=strategy_dict.get('feature_selection', 'all'),
        generation_passes=strategy_dict.get('generation_passes', 1),
        reward=reward,
        feedback=feedback,
        history=history_str,
        trend_analysis_section=trend_section,
        edge_effects_section=edge_effects_section
    )


# =============================================================================
# 科学家风格的反思模板 (Scientist-style Reflection Templates)
# =============================================================================

SCIENTIST_REFLECTION_TEMPLATE = """You are the Chief Scientist at SIGR Laboratory.
Your mission: Discover the optimal Gene Representation strategy through systematic experimentation.

## SYSTEM STATE
- Current Best Metric: {best_metric:.4f}
- Last Experiment Result: **{state}**
- Thinking Mode: **{thinking_mode}**
- Strategy Distance: {strategy_distance:.2f}

## EXPERIMENT FEEDBACK
{feedback_message}

## RECENT EXPERIMENT LOG
{experiment_log}
{edge_effects_section}
{kgbook_section}
## YOUR TASK

### Step 1: ANALYZE (Root Cause Analysis)
{analysis_instruction}

### Step 2: HYPOTHESIZE (Scientific Hypothesis)
Based on your analysis, formulate a NEW scientific hypothesis:
- "I hypothesize that [specific change] will [expected effect] because [reasoning]"
- Your hypothesis must be TESTABLE through parameter changes
- Be specific about which parameters you will modify and why

### Step 3: PLAN (Experiment Design)
Convert your hypothesis into concrete parameters.

Available parameters:
- edge_types: PPI, GO, HPO, TRRUST, CellMarker, Reactome
- sampling: top_k, random, weighted
- max_hops: 1-3
- max_neighbors: 10-200
- description_length: short (50-100 words), medium (100-150 words), long (150-250 words)
- description_focus: balanced, network, function, phenotype
- context_window: minimal, local, extended, full
- prompt_style: analytical, narrative, structured, comparative
- feature_selection: all, essential, diverse, task_specific
{trend_section}
## CONSTRAINTS
{mode_constraints}

## OUTPUT FORMAT
Provide your response as a JSON object:
```json
{{
    "hypothesis": "Your scientific hypothesis in one sentence",
    "expected_outcome": "What you expect to happen if hypothesis is correct",
    "reasoning": "Why you believe this will work based on evidence",
    "edge_types": [...],
    "max_hops": int,
    "sampling": "top_k" | "random" | "weighted",
    "max_neighbors": int,
    "description_length": "short" | "medium" | "long",
    "description_focus": "balanced" | "network" | "function" | "phenotype",
    "context_window": "minimal" | "local" | "extended" | "full",
    "prompt_style": "analytical" | "narrative" | "structured" | "comparative",
    "feature_selection": "all" | "essential" | "diverse" | "task_specific"
}}
```
"""

# Mode-specific constraints
MODE_CONSTRAINTS = {
    "FINE_TUNE": """
You are in **FINE-TUNE** mode (after breakthrough).
- Make SMALL adjustments only (< 20% parameter change)
- Preserve the core elements that led to success
- Goal: Consolidate and slightly improve the winning strategy
- Do NOT change edge_types significantly
- Do NOT make drastic changes to max_neighbors (±20 max)
""",
    "HIGH_ENTROPY": """
You are in **HIGH-ENTROPY** mode (STAGNATION detected!).
- You MUST make SIGNIFICANT changes (> 30% parameter difference)
- DO NOT repeat the previous strategy or similar variations
- Try completely different edge types, sampling methods, or description styles
- This is MANDATORY - small changes will be REJECTED and penalized
- Examples of significant changes:
  * Switch from top_k to weighted sampling
  * Add/remove multiple edge types
  * Change max_neighbors by ±50 or more
  * Switch description_focus entirely
""",
    "ANALYZE_AND_PIVOT": """
You are in **ANALYZE-AND-PIVOT** mode (exploration failed).
- Your previous hypothesis was invalidated
- Identify what went wrong and propose a DIFFERENT approach
- Avoid repeating the same mistake
- Medium-level changes recommended (20-30% parameter difference)
- Focus on why the hypothesis failed before proposing new one
"""
}

# Analysis instructions per state
ANALYSIS_INSTRUCTIONS = {
    "BREAKTHROUGH": """
Your last experiment was successful! Analyze:
- What made this hypothesis work?
- Which parameter changes contributed most to improvement?
- What should be preserved in future experiments?
""",
    "EXPLORATION_FAILURE": """
Your hypothesis was invalidated. Analyze:
- Why did this specific approach fail?
- Was the hypothesis fundamentally flawed or just the parameters?
- What alternative direction should we explore?
""",
    "STAGNATION": """
CRITICAL: You are stuck in a local optimum! Analyze:
- Why are you repeating similar strategies?
- What fundamental assumption might be wrong?
- What completely different approach have you NOT tried yet?
"""
}


def get_scientist_reflection_prompt(
    thinking_mode: str,
    reward_signal: 'RewardSignal',  # Forward reference
    experiment_history: List[Dict],
    task_name: str,
    current_strategy: Dict,
    best_metric: float = None,
    trend_analysis: Dict = None,
    edge_effects: Dict = None,
    kgbook_suggestions: str = None,
) -> str:
    """
    生成科学家风格的反思 prompt

    Args:
        thinking_mode: 思维模式 (FINE_TUNE / HIGH_ENTROPY / ANALYZE_AND_PIVOT)
        reward_signal: 结构化奖励信号
        experiment_history: 实验历史列表
        task_name: 任务名称
        current_strategy: 当前策略
        best_metric: 历史最佳指标
        trend_analysis: 趋势分析
        edge_effects: 边类型效果
        kgbook_suggestions: KGBOOK 建议

    Returns:
        str: 格式化的反思 prompt
    """
    # Format experiment log
    experiment_log = ""
    recent_experiments = experiment_history[-3:] if experiment_history else []
    for exp in recent_experiments:
        if hasattr(exp, 'to_experiment_report'):
            experiment_log += exp.to_experiment_report() + "\n\n"
        elif isinstance(exp, dict):
            exp_signal = exp.get('reward_signal', {})
            state = exp_signal.get('state', 'UNKNOWN') if exp_signal else 'UNKNOWN'
            metric = exp_signal.get('raw_metric', 0) if exp_signal else 0
            experiment_log += (
                f"## Experiment {exp.get('iteration', '?')}\n"
                f"**Hypothesis**: {exp.get('hypothesis', 'N/A')}\n"
                f"**Result**: {state} (metric={metric:.4f})\n"
                f"**Strategy**: edge_types={exp.get('strategy', {}).get('edge_types', [])}\n\n"
            )

    if not experiment_log:
        experiment_log = "No previous experiments recorded."

    # Get state and mode constraints
    state = reward_signal.state.value if reward_signal else "UNKNOWN"
    mode_constraints = MODE_CONSTRAINTS.get(thinking_mode, MODE_CONSTRAINTS["ANALYZE_AND_PIVOT"])
    analysis_instruction = ANALYSIS_INSTRUCTIONS.get(state, ANALYSIS_INSTRUCTIONS["EXPLORATION_FAILURE"])

    # Format trend section
    trend_section = ""
    if trend_analysis:
        trend_section = f"""
## TREND ANALYSIS
- Direction: {trend_analysis.get('trend_direction', 'unknown')}
- Strength: {trend_analysis.get('trend_strength', 0):.2f}
- Suggested Action: {trend_analysis.get('suggested_action', 'explore')}
"""

    # Format edge effects section
    edge_effects_section = ""
    if edge_effects:
        sorted_effects = sorted(
            edge_effects.items(),
            key=lambda x: x[1].get('ema_effect', 0),
            reverse=True
        )
        effects_lines = []
        for edge_type, effect in sorted_effects:
            usage = effect.get('usage_count', 0)
            success = effect.get('success_count', 0)
            success_rate = success / max(usage, 1)
            ema = effect.get('ema_effect', 0)
            effects_lines.append(
                f"  - {edge_type}: usage={usage}, success_rate={success_rate:.0%}, ema_effect={ema:+.4f}"
            )
        edge_effects_section = "\n## LEARNED EDGE EFFECTIVENESS (from Memory)\n" + "\n".join(effects_lines) + "\n"

    # Format kgbook suggestions
    kgbook_section = ""
    if kgbook_suggestions:
        kgbook_section = f"\n## KNOWLEDGE BASE SUGGESTIONS\n{kgbook_suggestions}\n"

    return SCIENTIST_REFLECTION_TEMPLATE.format(
        best_metric=best_metric or 0.0,
        state=state,
        thinking_mode=thinking_mode,
        strategy_distance=reward_signal.strategy_distance if reward_signal else 0.0,
        feedback_message=reward_signal.feedback_message if reward_signal else "No feedback available.",
        experiment_log=experiment_log,
        edge_effects_section=edge_effects_section,
        kgbook_section=kgbook_section,
        analysis_instruction=analysis_instruction,
        trend_section=trend_section,
        mode_constraints=mode_constraints,
    )


# =============================================================================
# 计算生物学家 Prompt (Computational Biologist Prompt)
# =============================================================================

BIO_SCIENTIST_PROMPT = """You are a Computational Biologist at the SIGR Laboratory.
Your mission: Discover optimal gene representation strategies through BIOLOGICAL REASONING.

## TASK CONTEXT: {task_name}
{task_biological_context}

## DOMAIN KNOWLEDGE
{domain_knowledge}

## SYSTEM STATE
- Current Best Metric: {best_metric:.4f}
- Last Experiment Result: **{state}**
- Thinking Mode: **{thinking_mode}**

## EXPERIMENT FEEDBACK
{feedback_message}

## HISTORICAL FACTS
{historical_facts}

{failure_patterns}

## YOUR ANALYSIS FRAMEWORK

### 1. BIOLOGICAL DIAGNOSIS
Analyze the last result from a BIOLOGICAL perspective:
- What biological mechanism might explain this outcome?
- How does the knowledge graph structure relate to the task?
- What signal-to-noise considerations apply?
- Think about WHY certain edge types work, not just IF they work

### 2. HYPOTHESIS FORMULATION
Propose a testable biological hypothesis:
"I hypothesize that [biological mechanism] will improve performance because [biological reasoning]"

Example hypotheses:
- "Reducing neighborhood size will improve cell type prediction because cell identity is defined by few marker genes, not the entire interactome"
- "Prioritizing HPO edges will improve dosage sensitivity prediction because haploinsufficient genes have specific phenotypic associations"
- "Switching to weighted sampling will better capture the heterogeneous importance of different interaction types"

### 3. EXPERIMENT DESIGN
Translate your hypothesis into parameters:

Available parameters:
- edge_types: PPI, GO, HPO, TRRUST, CellMarker, Reactome
- sampling: top_k, random, weighted
- max_hops: 1-3
- max_neighbors: 10-200
- description_length: short, medium, long
- description_focus: balanced, network, function, phenotype
- context_window: minimal, local, extended, full
- prompt_style: analytical, narrative, structured, comparative
- feature_selection: all, essential, diverse, task_specific

## CONSTRAINTS
{mode_constraints}

## OUTPUT FORMAT
Provide your response as a JSON object:
```json
{{{{
    "biological_diagnosis": "Your biological analysis of why the last experiment succeeded/failed",
    "hypothesis": "Your biological hypothesis",
    "expected_mechanism": "How this will work biologically",
    "edge_types": [...],
    "max_hops": int,
    "sampling": "top_k" | "random" | "weighted",
    "max_neighbors": int,
    "description_length": "short" | "medium" | "long",
    "description_focus": "balanced" | "network" | "function" | "phenotype",
    "context_window": "minimal" | "local" | "extended" | "full",
    "prompt_style": "analytical" | "narrative" | "structured" | "comparative",
    "feature_selection": "all" | "essential" | "diverse" | "task_specific"
}}}}
```
"""


def get_biologist_reflection_prompt(
    thinking_mode: str,
    reward_signal: 'RewardSignal',
    task_name: str,
    current_strategy: Dict,
    task_biological_context: str,
    domain_knowledge: str,
    historical_facts: str,
    failure_patterns: str = "",
    best_metric: float = None,
) -> str:
    """
    生成计算生物学家风格的反思 prompt

    核心改变：
    1. 不注入 UCB 建议
    2. 注入生物学背景知识
    3. 要求生物学层面的诊断和假设

    Args:
        thinking_mode: 思维模式
        reward_signal: 结构化奖励信号
        task_name: 任务名称
        current_strategy: 当前策略
        task_biological_context: 任务特定的生物学背景
        domain_knowledge: 通用领域知识
        historical_facts: 历史事实（非建议）
        failure_patterns: 失败模式分析
        best_metric: 历史最佳指标

    Returns:
        str: 格式化的反思 prompt
    """
    # Get state and mode constraints
    state = reward_signal.state.value if reward_signal else "UNKNOWN"
    mode_constraints = MODE_CONSTRAINTS.get(thinking_mode, MODE_CONSTRAINTS["ANALYZE_AND_PIVOT"])

    return BIO_SCIENTIST_PROMPT.format(
        task_name=task_name,
        task_biological_context=task_biological_context,
        domain_knowledge=domain_knowledge,
        best_metric=best_metric or 0.0,
        state=state,
        thinking_mode=thinking_mode,
        feedback_message=reward_signal.feedback_message if reward_signal else "No feedback available.",
        historical_facts=historical_facts,
        failure_patterns=failure_patterns,
        mode_constraints=mode_constraints,
    )


# =============================================================================
# Bio-CoT Prompt：强制生物学思维链推理
# =============================================================================

# 任务特定的诊断问题 (Task-Specific Diagnosis Questions)
TASK_DIAGNOSIS_QUESTIONS = {
    'cell': """
- Is cell identity being captured by the current edge types?
- Are marker genes (CellMarker) being prioritized over generic protein interactions?
- Is the neighborhood size appropriate for marker-defined cell identity?
- Are tissue-specific expression patterns being captured?
""",

    'geneattribute_dosage_sensitivity': """
- Are hub genes (haploinsufficient) being properly represented?
- Is regulatory network position captured through TRRUST edges?
- Are phenotype associations (HPO) being leveraged for dosage effects?
- Is evolutionary constraint information being captured (pLI, LOEUF context)?
""",

    'geneattribute_bivalent': """
- Are developmental and pluripotency-related annotations captured?
- Is Polycomb (PRC2) regulatory context being represented?
- Are lineage commitment genes being distinguished?
- Is the balance between active (H3K4me3) and repressive (H3K27me3) marks reflected?
""",

    'geneattribute_lys4_only': """
- Are active promoter characteristics being captured?
- Is housekeeping gene context being represented?
- Are constitutive expression patterns reflected in the representation?
""",

    'geneattribute_no_methylation': """
- Are CpG island associations being captured?
- Is promoter accessibility context being represented?
- Are open chromatin characteristics reflected?
""",

    'ppi': """
- Are protein localization constraints being captured?
- Is functional similarity (GO) complementing physical interaction evidence?
- Are interaction partners being properly contextualized within pathways?
- Is hub protein bias being addressed?
""",

    'ggi': """
- Are genetic interaction patterns (epistasis) being captured?
- Is functional redundancy between paralogs being represented?
- Are shared pathway memberships being leveraged?
""",

    'genetype': """
- Are molecular function annotations (GO) being properly weighted?
- Is enzyme/receptor/transporter classification information being captured?
- Are catalytic domain features being represented?
""",

    'perturbation': """
- Are regulatory cascade effects (TRRUST) being captured?
- Is signal propagation through the network being modeled?
- Are direct vs. indirect effects being distinguished?
""",
}


BIO_COT_PROMPT = """You are a Computational Biologist conducting systematic scientific experiments.
Your goal is NOT to tune parameters, but to discover biological mechanisms.

## SCIENTIFIC METHOD FRAMEWORK

You MUST follow this exact reasoning chain. Do NOT skip steps.

### STEP 1: OBSERVATION (What happened?)
State the experimental facts objectively:
- Previous metric: {prev_metric}
- Current metric: {curr_metric}
- Change: {delta} ({experiment_state})
- Strategy used: {strategy_summary}

### STEP 2: BIOLOGICAL DIAGNOSIS (Why did this happen?)
Analyze from a BIOLOGICAL perspective. Think about:
{task_diagnosis_questions}

Key questions:
- What biological mechanism might explain this outcome?
- How does the knowledge graph structure relate to the task?
- What signal-to-noise considerations apply?

### STEP 3: HYPOTHESIS FORMULATION (What should we try?)
Based on your diagnosis, propose a TESTABLE biological hypothesis.

Format: "I hypothesize that [biological mechanism] will [expected effect] because [biological reasoning]"

Your hypothesis MUST be:
- **BIOLOGICAL**: About mechanisms, not parameters (BAD: "increase max_neighbors", GOOD: "focus on marker genes")
- **TESTABLE**: Can be validated/falsified by the next experiment
- **FALSIFIABLE**: Has clear failure criteria

### STEP 4: EXPERIMENT DESIGN (How do we test it?)
ONLY NOW translate your biological hypothesis into experimental parameters.
Each parameter choice MUST logically follow from your hypothesis.

Explain the connection:
- Hypothesis: "cell identity is marker-defined" → edge_types=['CellMarker', 'GO'] (because markers directly define identity)
- Hypothesis: "fewer genes define cell type" → max_neighbors=20 (because cell identity is sparse, not dense)

Available parameters:
- edge_types: PPI, GO, HPO, TRRUST, CellMarker, Reactome
- sampling: top_k (prioritize strongest), random (explore), weighted (balance)
- max_hops: 1-3 (network depth)
- max_neighbors: 10-200 (receptive field size)
- description_length: short (50-100 words), medium (100-150), long (150-250)
- description_focus: balanced, network (hub/centrality), function (pathways), phenotype (disease)
- context_window: minimal (direct), local (1-hop), extended (2-hop), full
- prompt_style: analytical, narrative, structured, comparative
- feature_selection: all, essential (core only), diverse (maximize variety), task_specific

### STEP 5: FALSIFICATION CRITERIA (How will we know if we're wrong?)
Define clear conditions that would invalidate your hypothesis:
- If metric < X, then the hypothesis about [Y] is wrong
- If [specific pattern], then the biological assumption about [Z] was incorrect

## TASK CONTEXT: {task_name}
{task_biological_context}

## HYPOTHESIS LEDGER (Your accumulated scientific knowledge)
{hypothesis_ledger_summary}

{failure_patterns}

## CONSTRAINTS
{mode_constraints}

## OUTPUT FORMAT
Provide your response as a JSON object. The structure MUST match exactly:

```json
{{{{
    "observation": {{{{
        "previous_metric": <float>,
        "current_metric": <float>,
        "delta": <float>,
        "state": "<BREAKTHROUGH|EXPLORATION_FAILURE|STAGNATION>"
    }}}},
    "biological_diagnosis": "<Your biological analysis of why this happened>",
    "hypothesis": {{{{
        "statement": "<I hypothesize that...>",
        "biological_basis": "<The biological mechanism is...>",
        "expected_outcome": "<This should result in...>"
    }}}},
    "experiment_design": {{{{
        "rationale": "<How parameters derive from hypothesis>",
        "edge_types": [<list of edge types>],
        "max_hops": <int>,
        "sampling": "<top_k|random|weighted>",
        "max_neighbors": <int>,
        "description_length": "<short|medium|long>",
        "description_focus": "<balanced|network|function|phenotype>",
        "context_window": "<minimal|local|extended|full>",
        "prompt_style": "<analytical|narrative|structured|comparative>",
        "feature_selection": "<all|essential|diverse|task_specific>"
    }}}},
    "falsification_criteria": "<If..., then hypothesis is wrong>"
}}}}
```
"""


def get_bio_cot_prompt(
    task_name: str,
    task_biological_context: str,
    prev_metric: float,
    curr_metric: float,
    experiment_state: str,
    strategy_summary: str,
    hypothesis_ledger_summary: str,
    failure_patterns: str = "",
    mode_constraints: str = "",
) -> str:
    """
    生成 Bio-CoT (Biological Chain of Thought) Prompt

    这个 prompt 强制 Actor 按照科学方法进行推理：
    Observation → Diagnosis → Hypothesis → Design → Falsification

    Args:
        task_name: 任务名称
        task_biological_context: 任务特定的生物学背景
        prev_metric: 上一次实验的指标
        curr_metric: 当前实验的指标
        experiment_state: 实验状态 (BREAKTHROUGH/EXPLORATION_FAILURE/STAGNATION)
        strategy_summary: 当前策略摘要
        hypothesis_ledger_summary: 假设账本摘要
        failure_patterns: 失败模式分析
        mode_constraints: 思维模式约束

    Returns:
        str: 格式化的 Bio-CoT prompt
    """
    # 获取任务特定的诊断问题
    task_diagnosis_questions = TASK_DIAGNOSIS_QUESTIONS.get(
        task_name,
        "- What biological mechanisms are relevant to this task?\n- What signal sources should be prioritized?"
    )

    # 计算 delta
    delta = curr_metric - prev_metric if prev_metric else curr_metric
    delta_str = f"{delta:+.4f}" if prev_metric else "N/A (first experiment)"

    # 格式化指标
    prev_metric_str = f"{prev_metric:.4f}" if prev_metric else "N/A"
    curr_metric_str = f"{curr_metric:.4f}"

    return BIO_COT_PROMPT.format(
        task_name=task_name,
        task_biological_context=task_biological_context,
        prev_metric=prev_metric_str,
        curr_metric=curr_metric_str,
        delta=delta_str,
        experiment_state=experiment_state,
        strategy_summary=strategy_summary,
        task_diagnosis_questions=task_diagnosis_questions,
        hypothesis_ledger_summary=hypothesis_ledger_summary,
        failure_patterns=failure_patterns,
        mode_constraints=mode_constraints,
    )


def format_strategy_summary(strategy: Dict) -> str:
    """
    格式化策略摘要

    Args:
        strategy: 策略字典

    Returns:
        str: 人类可读的策略摘要
    """
    edge_types = strategy.get('edge_types', [])
    max_neighbors = strategy.get('max_neighbors', 50)
    sampling = strategy.get('sampling', 'top_k')
    max_hops = strategy.get('max_hops', 2)
    desc_length = strategy.get('description_length', 'medium')

    return (
        f"edge_types={edge_types}, max_neighbors={max_neighbors}, "
        f"sampling={sampling}, max_hops={max_hops}, description_length={desc_length}"
    )
