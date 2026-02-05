"""
Biological Domain Knowledge for SIGR Actor

This module contains task-specific biological context and domain knowledge
that helps the Actor reason like a Computational Biologist rather than a
parameter tuning algorithm.

Key Design Principles:
- Provide biological context, not parameter suggestions
- Help LLM understand WHY certain approaches work for specific tasks
- Enable hypothesis-driven reasoning based on biological mechanisms
"""

# Task-specific biological context
TASK_BIOLOGICAL_CONTEXT = {
    'geneattribute_dosage_sensitivity': """
**Dosage Sensitivity Prediction**

Biological Context:
- Dosage-sensitive genes often encode hub proteins in protein complexes
- Haploinsufficiency correlates with evolutionary constraint (pLI, LOEUF scores)
- These genes are frequently transcription factors or signaling pathway components
- Loss of one copy leads to phenotypic consequences due to stoichiometric imbalance

Key Signal Sources:
- HPO: Phenotypes associated with gene dosage changes (high signal)
- TRRUST: Regulatory relationships indicating network position
- PPI: Hub status in protein interaction network
- Reactome: Pathway membership may indicate functional importance

Biological Considerations:
- Hub genes have many PPI edges but not all are dosage-sensitive
- Regulatory targets (TRRUST) may indicate downstream effects
- GO terms like "protein complex" or "transcription factor" are informative
- HPO terms directly capture dosage-related phenotypes

Warning Signs:
- Too many neighbors -> noise from peripheral interactions
- Random sampling -> may miss critical regulatory edges
- Ignoring HPO -> losing direct phenotype signal
""",

    'geneattribute_lys4_only': """
**Lys4-Only Methylation Prediction**

Biological Context:
- H3K4 methylation marks active promoters and enhancers
- Lys4-only genes lack H3K27me3 (not bivalent)
- Often associated with housekeeping genes or lineage-specific active genes
- Related to transcriptional activation and chromatin accessibility

Key Signal Sources:
- GO: Chromatin modification, transcription regulation terms
- Reactome: Epigenetic pathways, gene expression regulation
- PPI: Interactions with chromatin modifiers (MLL complex, etc.)

Biological Considerations:
- Focus on chromatin-related annotations
- Transcription factor interactions are informative
- Avoid over-weighting generic PPI edges

Warning Signs:
- Too broad neighborhood dilutes epigenetic signal
- Missing GO chromatin terms loses key information
""",

    'geneattribute_no_methylation': """
**No Methylation Prediction**

Biological Context:
- Genes without H3K4 or H3K27 methylation
- Often in heterochromatin or silenced regions
- May be tissue-specific silent genes or non-coding regions
- Less transcriptionally active

Key Signal Sources:
- GO: Look for absence of transcription-related terms
- HPO: May lack phenotype associations
- PPI: May have fewer interactions

Biological Considerations:
- Negative signal is important (what's NOT there)
- Sparse connectivity may be informative
- Consider gene expression context

Warning Signs:
- Over-connecting through PPI may introduce noise
""",

    'geneattribute_bivalent': """
**Bivalent Chromatin Prediction**

Biological Context:
- Bivalent promoters carry both H3K4me3 (active) and H3K27me3 (repressive) marks
- Characteristic of developmental/pluripotency genes
- "Poised" for activation upon differentiation signals
- Often transcription factors controlling cell fate decisions

Key Signal Sources:
- GO: Development, differentiation, morphogenesis terms
- HPO: Developmental phenotypes
- CellMarker: Stem cell and progenitor markers
- TRRUST: Regulatory networks of developmental TFs

Biological Considerations:
- Focus on developmental and pluripotency-related annotations
- Transcription factor annotations are highly informative
- CellMarker data for stem/progenitor cells is valuable

Warning Signs:
- Generic PPI edges may dilute developmental signal
- Too many housekeeping gene connections add noise
""",

    'ppi': """
**Protein-Protein Interaction Prediction**

Biological Context:
- Goal is to predict whether two proteins physically interact
- Interacting proteins often share:
  - Subcellular localization
  - Functional annotations (GO)
  - Pathway membership
  - Co-expression patterns

Key Signal Sources:
- GO: Cellular component (localization) and biological process
- Reactome: Pathway co-membership
- Existing PPI: Network structure/topology

Biological Considerations:
- Proteins in same compartment more likely to interact
- Shared pathway membership is predictive
- Network topology (common neighbors) is informative
- Hub proteins have many interactions

Warning Signs:
- Using only PPI edges creates circularity
- Ignoring localization loses physical constraint information
""",

    'genetype': """
**Gene Type Classification**

Biological Context:
- Classifying genes by functional type (e.g., protein-coding, lncRNA, etc.)
- Different gene types have distinct:
  - Regulatory patterns
  - Network positions
  - Functional annotations

Key Signal Sources:
- GO: Molecular function annotations
- TRRUST: Regulatory relationships
- Reactome: Pathway involvement

Biological Considerations:
- Protein-coding genes have richer functional annotations
- lncRNAs may lack GO annotations but have regulatory roles
- Gene type correlates with network connectivity

Warning Signs:
- Missing genes without GO annotations biases prediction
- Annotation bias affects certain gene types
""",

    'ggi': """
**Gene-Gene Interaction Prediction**

Biological Context:
- Genetic interactions (epistasis) differ from physical interactions
- Genes interact genetically when combined perturbations show non-additive effects
- Often indicates:
  - Parallel pathways
  - Functional redundancy
  - Regulatory relationships

Key Signal Sources:
- GO: Shared biological process (functional similarity)
- Reactome: Pathway relationships
- TRRUST: Regulatory connections
- HPO: Shared phenotypes

Biological Considerations:
- Genetic interaction != physical interaction
- Paralogs often show genetic interactions
- Pathway structure is informative

Warning Signs:
- PPI alone misses functional relationships
- Need broader functional context than physical interactions
""",

    'cell': """
**Cell Type Classification**

Biological Context:
- Cell identity is defined by a small set of marker genes
- Marker genes show tissue-specific expression patterns
- Single-cell data reveals cell type clustering based on few genes
- Cell types form hierarchies (e.g., immune -> T cell -> CD8+)

Key Signal Sources:
- CellMarker: Curated cell type marker genes (HIGHEST PRIORITY)
- GO: Biological process annotations for cell function
- HPO: Disease associations may indicate cell type relevance

Biological Considerations:
- Cell identity comes from few specific genes, not genome-wide patterns
- Marker specificity > marker quantity
- Tissue context matters (same gene, different cell types)
- Hierarchical relationships between cell types

Warning Signs:
- Large neighborhoods dilute cell-specific signals
- Generic edges (PPI) may introduce non-specific noise
- Missing CellMarker edges loses primary signal
- Random sampling may miss rare marker connections
""",

    'perturbation': """
**Perturbation Response Prediction**

Biological Context:
- Predicting gene expression changes after perturbation
- Requires understanding of:
  - Regulatory networks (direct effects)
  - Pathway propagation (indirect effects)
  - Network topology (signal spread)

Key Signal Sources:
- TRRUST: Regulatory relationships (HIGHEST PRIORITY - causal)
- Reactome: Pathway membership for effect propagation
- PPI: Signal transduction networks
- GO: Functional context

Biological Considerations:
- Regulatory edges are directional and causal
- Effect propagation follows network structure
- Hub genes amplify perturbation effects
- Pathway context determines response direction

Warning Signs:
- Ignoring TRRUST loses causal regulatory information
- Too many hops spreads signal too thin
- Missing pathway context loses propagation logic
"""
}


# General domain knowledge about the knowledge graph
DOMAIN_KNOWLEDGE = """
## Knowledge Graph Biology

### Edge Type Semantics and Signal Quality:

| Edge Type | Description | Signal Quality | Connectivity |
|-----------|-------------|----------------|--------------|
| PPI | Physical protein interactions | Medium (noisy) | High (dense) |
| GO | Functional annotations | High (curated) | Medium |
| HPO | Phenotype associations | Very High | Low (sparse) |
| TRRUST | Regulatory relationships | Very High (causal) | Low |
| CellMarker | Cell type markers | Very High (specific) | Low |
| Reactome | Pathway membership | High | Medium |

### Signal-to-Noise Principles:

1. **Hub Gene Bias**: Hub genes have many PPI edges, which can dominate embeddings.
   - Solution: Weighted sampling or neighbor limits

2. **Annotation Bias**: Well-studied genes have more annotations.
   - Solution: Consider edge source diversity

3. **Hierarchical Redundancy**: GO terms have parent-child relationships.
   - Solution: Prefer specific (child) terms over general (parent) terms

4. **Sparsity vs. Specificity**: Sparse edges (HPO, CellMarker) are often more specific.
   - Solution: Weight sparse edges higher, don't ignore them in sampling

5. **Directionality**: TRRUST regulatory edges are directional (TF -> target).
   - Solution: Consider edge direction when relevant to task

### Neighborhood Size Intuition:

- **Small (10-30)**: Best for tasks where few specific genes define the signal
  - Cell type classification, dosage sensitivity

- **Medium (30-70)**: Best for tasks requiring functional context
  - PPI prediction, gene type classification

- **Large (70-150)**: Best for tasks requiring broad network context
  - Perturbation prediction, pathway analysis

### Hop Distance Intuition:

- **1-hop**: Direct relationships only. High signal, limited context.
- **2-hop**: Include neighbors of neighbors. Good balance.
- **3-hop**: Broad context but may include functionally unrelated genes.

### Sampling Strategy Intuition:

- **top_k**: Prioritizes high-connectivity edges. Good for well-annotated genes.
- **weighted**: Balances edge types. Good for diverse signal sources.
- **random**: Maximum exploration. Use when stuck in local optima.
"""


def get_task_biological_context(task_name: str) -> str:
    """
    Get biological context for a specific task.

    Args:
        task_name: Name of the task

    Returns:
        Biological context string
    """
    return TASK_BIOLOGICAL_CONTEXT.get(task_name, "No specific biological context available.")


def get_domain_knowledge() -> str:
    """
    Get general domain knowledge about the knowledge graph.

    Returns:
        Domain knowledge string
    """
    return DOMAIN_KNOWLEDGE


def get_full_biological_context(task_name: str) -> str:
    """
    Get combined task-specific and general biological context.

    Args:
        task_name: Name of the task

    Returns:
        Combined context string
    """
    task_context = get_task_biological_context(task_name)
    return f"{task_context}\n\n{DOMAIN_KNOWLEDGE}"
