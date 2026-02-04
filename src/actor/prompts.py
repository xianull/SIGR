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

from typing import Dict, List


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
- EXPLORE: Try significantly different parameters (plateau/declining trend)
- FINE-TUNE: Make small adjustments to current approach (improving trend)
- PRESERVE: Keep successful elements, only fix weaknesses

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

Based on Steps 1-3, propose improvements:
- What specific parameter(s) to change?
- Why this change addresses the root cause?
- Expected effect of the change?

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
    best_reward: float = None
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

    return REFLECTION_COT_PROMPT_TEMPLATE.format(
        task_name=task_name,
        edge_types=strategy_dict.get('edge_types', []),
        max_hops=strategy_dict.get('max_hops', 2),
        sampling=strategy_dict.get('sampling', 'top_k'),
        max_neighbors=strategy_dict.get('max_neighbors', 50),
        description_length=strategy_dict.get('description_length', 'medium'),
        edge_weights=edge_weights if edge_weights else 'Not set',
        neighbors_per_type=neighbors_per_type if neighbors_per_type else 'Not set',
        include_statistics=strategy_dict.get('include_statistics', True),
        focus_keywords=focus_keywords if focus_keywords else 'Not set',
        reward=reward,
        feedback=feedback,
        history=history_str,
        trend_analysis_section=trend_section
    )
