"""
Task-Specific Prompts and Edge Priorities for SIGR Actor

Defines initial prompts and edge type priorities for each downstream task.
All prompts are in English with anti-leakage constraints.
"""

from typing import Dict, List


# Task-specific edge type priorities
# These define which edge types are most relevant for each task
TASK_EDGE_PRIORITIES: Dict[str, List[str]] = {
    'ppi': ['PPI', 'GO', 'Reactome', 'TRRUST'],  # Protein interactions are primary
    'genetype': ['GO', 'Reactome', 'PPI', 'HPO'],  # Gene Ontology is primary
    'phenotype': ['HPO', 'GO', 'Reactome', 'PPI'],  # Phenotype associations are primary
    'celltype': ['CellMarker', 'GO', 'Reactome', 'PPI'],  # Cell type markers are primary
    'dosage': ['TRRUST', 'GO', 'Reactome', 'PPI'],  # Regulatory relations are primary
    # GGI task
    'ggi': ['PPI', 'GO', 'Reactome', 'TRRUST'],  # Gene-gene interaction
    # Cell task
    'cell': ['CellMarker', 'GO', 'Reactome', 'PPI'],  # Cell type classification
    # GeneAttribute subtasks
    'geneattribute_dosage_sensitivity': ['TRRUST', 'GO', 'HPO', 'Reactome', 'PPI'],  # Gene essentiality + phenotype + pathway
    'geneattribute_lys4_only': ['GO', 'Reactome', 'TRRUST', 'HPO'],  # Chromatin/methylation
    'geneattribute_no_methylation': ['GO', 'Reactome', 'TRRUST', 'HPO'],  # Chromatin/methylation
    'geneattribute_bivalent': ['GO', 'Reactome', 'TRRUST', 'HPO'],  # Chromatin state
}


# Task-specific initial prompts
# These are the starting prompts that will be optimized by the Actor
TASK_INITIAL_PROMPTS: Dict[str, str] = {
    'ppi': """Generate a biological description for gene {gene_id} ({gene_name}) to support protein-protein interaction prediction.

Knowledge Graph Information:
- Protein Interactions: {ppi_info}
- Biological Processes: {go_info}
- Pathway Involvement: {pathway_info}
- Regulatory Relations: {tf_info}

Focus on:
- Protein domains and binding sites
- Signaling pathway involvement
- Known interaction partners
- Subcellular localization

IMPORTANT CONSTRAINTS:
- Only describe known biological information
- Do NOT make any predictions or judgments
- Do NOT output classification labels
- Do NOT assess probabilities or likelihood
- Description must be factual, not speculative
""",

    'genetype': """Generate a biological description for gene {gene_id} ({gene_name}) to support gene functional type classification.

Knowledge Graph Information:
- Biological Processes: {go_info}
- Molecular Functions: {function_info}
- Pathway Involvement: {pathway_info}

Focus on:
- Molecular function (enzymatic activity, binding capability, etc.)
- Biological processes involved
- Metabolic pathway roles
- Cellular component localization

IMPORTANT CONSTRAINTS:
- Only describe functional information
- Do NOT output classification labels
- Do NOT predict gene type categories
- Description must be factual, not speculative
""",

    'phenotype': """Generate a biological description for gene {gene_id} ({gene_name}) to support gene-phenotype association prediction.

Knowledge Graph Information:
- Phenotype Associations: {phenotype_info}
- Disease Links: {disease_info}
- Biological Processes: {go_info}

Focus on:
- Known disease associations
- Mutation-induced phenotypes
- Clinical symptom correlations
- Phenotypic features in model organisms

IMPORTANT CONSTRAINTS:
- Only describe known information
- Do NOT predict unknown phenotype associations
- Do NOT output labels or categories
- Description must be factual, not speculative
""",

    'celltype': """Generate a biological description for gene {gene_id} ({gene_name}) to support cell type marker prediction.

Knowledge Graph Information:
- Cell Type Markers: {celltype_info}
- Tissue Expression: {tissue_info}
- Biological Processes: {go_info}

Focus on:
- Tissue-specific expression patterns
- Cell differentiation markers
- Developmental stage expression
- Single-cell expression profiles

IMPORTANT CONSTRAINTS:
- Only describe expression patterns
- Do NOT predict whether gene is a marker for specific cell types
- Do NOT output classification labels
- Description must be factual, not speculative
""",

    'dosage': """Generate a biological description for gene {gene_id} ({gene_name}) to support dosage sensitivity prediction.

Knowledge Graph Information:
- Regulatory Relations: {tf_info}
- Biological Processes: {go_info}
- Interaction Network: {ppi_info}

Focus on:
- Essentiality (whether required for survival)
- Position in regulatory networks
- Genomic constraint level
- Haploinsufficiency evidence

IMPORTANT CONSTRAINTS:
- Only describe gene characteristics
- Do NOT predict sensitivity categories
- Do NOT output classification labels
- Description must be factual, not speculative
""",

    # GGI task
    'ggi': """Generate a biological description for gene {gene_id} ({gene_name}) to support gene-gene interaction prediction.

Knowledge Graph Information:
- Protein Interactions: {ppi_info}
- Biological Processes: {go_info}
- Pathway Involvement: {pathway_info}
- Regulatory Relations: {tf_info}

Focus on:
- Known interaction partners and pathways
- Co-expression patterns
- Shared biological processes
- Regulatory relationships

IMPORTANT CONSTRAINTS:
- Only describe known biological information
- Do NOT predict specific interactions
- Do NOT output classification labels
- Description must be factual, not speculative
""",

    # Cell task
    'cell': """Generate a biological description for gene {gene_id} ({gene_name}) to support cell type classification.

Knowledge Graph Information:
- Cell Type Markers: {celltype_info}
- Biological Processes: {go_info}
- Protein Interactions: {ppi_info}

Focus on:
- Cell type specific expression patterns
- Marker gene characteristics
- Tissue distribution
- Developmental expression

IMPORTANT CONSTRAINTS:
- Only describe expression patterns
- Do NOT predict cell type associations
- Do NOT output classification labels
- Description must be factual, not speculative
""",

    # GeneAttribute subtasks
    'geneattribute_dosage_sensitivity': """Describe the biological characteristics of gene {gene_name} ({gene_id}).

Available information:
- Regulatory network: {tf_info}
- Biological processes: {go_info}
- Phenotype associations: {phenotype_info}
- Pathway involvement: {pathway_info}
- Protein interactions: {ppi_info}

Write a natural, flowing description that covers:
- Evidence of gene essentiality or functional importance
- Position in regulatory and interaction networks
- Known phenotypic consequences of gene perturbation
- Evolutionary constraint or conservation evidence

CONSTRAINTS:
- Use factual information only
- DO NOT predict dosage sensitivity
- DO NOT output classification labels
- Vary your writing style based on available data
""",

    'geneattribute_lys4_only': """Generate a biological description for gene {gene_id} ({gene_name}) to support chromatin state prediction.

Knowledge Graph Information:
- Biological Processes: {go_info}
- Regulatory Relations: {tf_info}
- Phenotype Associations: {phenotype_info}

Focus on:
- Transcriptional regulation patterns
- Epigenetic context
- Gene expression characteristics
- Chromatin accessibility indicators

IMPORTANT CONSTRAINTS:
- Only describe gene characteristics
- Do NOT predict methylation states
- Do NOT output classification labels
- Description must be factual, not speculative
""",

    'geneattribute_no_methylation': """Generate a biological description for gene {gene_id} ({gene_name}) to support chromatin state prediction.

Knowledge Graph Information:
- Biological Processes: {go_info}
- Regulatory Relations: {tf_info}
- Phenotype Associations: {phenotype_info}

Focus on:
- Transcriptional activity patterns
- Promoter characteristics
- Expression regulation
- Chromatin environment

IMPORTANT CONSTRAINTS:
- Only describe gene characteristics
- Do NOT predict methylation states
- Do NOT output classification labels
- Description must be factual, not speculative
""",

    'geneattribute_bivalent': """Generate a biological description for gene {gene_id} ({gene_name}) to support bivalent chromatin state prediction.

Knowledge Graph Information:
- Biological Processes: {go_info}
- Regulatory Relations: {tf_info}
- Phenotype Associations: {phenotype_info}

Focus on:
- Developmental gene characteristics
- Pluripotency associations
- Transcriptional poising
- Lineage commitment roles

IMPORTANT CONSTRAINTS:
- Only describe gene characteristics
- Do NOT predict chromatin states
- Do NOT output classification labels
- Description must be factual, not speculative
""",
}


# Template for the Actor's reflection prompt
REFLECTION_PROMPT_TEMPLATE = """You are optimizing a gene representation strategy for the {task_name} task.

Current Strategy:
- Edge types: {edge_types}
- Max hops: {max_hops}
- Sampling method: {sampling}
- Max neighbors: {max_neighbors}

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

Available edge types: PPI, GO, HPO, TRRUST, CellMarker, Reactome
Available sampling methods: top_k, random, weighted
Max hops range: 1-3
Max neighbors range: 10-200

Output a JSON with the updated strategy:
{{
    "edge_types": [...],
    "max_hops": int,
    "sampling": "top_k" | "random" | "weighted",
    "max_neighbors": int,
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

    return REFLECTION_PROMPT_TEMPLATE.format(
        task_name=task_name,
        edge_types=strategy_dict.get('edge_types', []),
        max_hops=strategy_dict.get('max_hops', 2),
        sampling=strategy_dict.get('sampling', 'top_k'),
        max_neighbors=strategy_dict.get('max_neighbors', 50),
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

Based on Steps 1-3, propose improvements:
- What specific parameter(s) to change?
- Why this change addresses the root cause?
- Expected effect of the change?

## STEP 5: OUTPUT STRATEGY
Available options:
- Edge types: PPI, GO, HPO, TRRUST, CellMarker, Reactome
- Sampling: top_k, random, weighted
- Max hops: 1-3
- Max neighbors: 10-200

```json
{{
    "edge_types": [...],
    "max_hops": int,
    "sampling": "top_k" | "random" | "weighted",
    "max_neighbors": int,
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

    return REFLECTION_COT_PROMPT_TEMPLATE.format(
        task_name=task_name,
        edge_types=strategy_dict.get('edge_types', []),
        max_hops=strategy_dict.get('max_hops', 2),
        sampling=strategy_dict.get('sampling', 'top_k'),
        max_neighbors=strategy_dict.get('max_neighbors', 50),
        reward=reward,
        feedback=feedback,
        history=history_str,
        trend_analysis_section=trend_section
    )
