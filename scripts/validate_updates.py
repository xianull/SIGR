#!/usr/bin/env python
"""
Validation script for SIGR framework updates.

Verifies that all P0, P1, P3 changes are correctly integrated.
"""

import os
import sys
from pathlib import Path

# Set up proper paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Change to project root for relative imports to work
os.chdir(project_root)


def test_strategy_config():
    """Test StrategyConfig with new parameters."""
    from src.actor.strategy import StrategyConfig, Strategy, DESCRIPTION_LENGTH_WORDS

    # Test new parameters
    config = StrategyConfig(
        edge_types=['PPI', 'GO', 'OMIM', 'GTEx', 'CORUM'],
        max_hops=2,
        sampling='top_k',
        max_neighbors=50,
        description_length='medium',
        edge_weights={'PPI': 1.0, 'GO': 0.8},
        neighbors_per_type={'PPI': 30, 'GO': 20},
        include_statistics=True,
        focus_keywords=['hub', 'conserved', 'essential']
    )

    assert config.description_length == 'medium'
    assert config.edge_weights == {'PPI': 1.0, 'GO': 0.8}
    assert config.neighbors_per_type == {'PPI': 30, 'GO': 20}
    assert config.include_statistics == True
    assert config.focus_keywords == ['hub', 'conserved', 'essential']

    # Test new edge types
    strategy = Strategy(config)
    assert 'OMIM' in strategy.VALID_EDGE_TYPES
    assert 'GTEx' in strategy.VALID_EDGE_TYPES
    assert 'CORUM' in strategy.VALID_EDGE_TYPES

    print("✓ StrategyConfig tests passed")


def test_prompts():
    """Test prompt templates and implicit signals."""
    from src.actor.prompts import (
        TASK_INITIAL_PROMPTS,
        TASK_IMPLICIT_SIGNALS,
        TASK_EDGE_PRIORITIES
    )

    # Verify all tasks have prompts
    tasks = [
        'ppi', 'genetype', 'phenotype', 'celltype', 'dosage',
        'ggi', 'cell',
        'geneattribute_dosage_sensitivity',
        'geneattribute_lys4_only',
        'geneattribute_no_methylation',
        'geneattribute_bivalent'
    ]

    for task in tasks:
        assert task in TASK_INITIAL_PROMPTS, f"Missing prompt for {task}"
        assert task in TASK_IMPLICIT_SIGNALS, f"Missing implicit signals for {task}"
        assert task in TASK_EDGE_PRIORITIES, f"Missing edge priorities for {task}"

    # Verify word limit in prompts
    for task, prompt in TASK_INITIAL_PROMPTS.items():
        assert '100-150 words' in prompt or '150 words' in prompt, \
            f"Missing word limit in {task} prompt"

    # Verify new edge types in priorities
    assert 'OMIM' in TASK_EDGE_PRIORITIES.get('phenotype', [])
    assert 'GTEx' in TASK_EDGE_PRIORITIES.get('celltype', [])
    assert 'CORUM' in TASK_EDGE_PRIORITIES.get('ppi', [])

    print("✓ Prompts tests passed")


def test_kg_utils():
    """Test KG utilities with new edge types."""
    from src.utils.kg_utils import EDGE_TYPE_MAPPING, EDGE_LABEL_MAPPING

    # Verify new edge types are mapped
    assert 'gene to disease association' in EDGE_TYPE_MAPPING
    assert 'expressed in tissue' in EDGE_TYPE_MAPPING
    assert 'in complex' in EDGE_TYPE_MAPPING

    assert EDGE_TYPE_MAPPING['gene to disease association'] == 'OMIM'
    assert EDGE_TYPE_MAPPING['expressed in tissue'] == 'GTEx'
    assert EDGE_TYPE_MAPPING['in complex'] == 'CORUM'

    # Verify reverse mapping
    assert EDGE_LABEL_MAPPING['OMIM'] == 'gene to disease association'
    assert EDGE_LABEL_MAPPING['GTEx'] == 'expressed in tissue'
    assert EDGE_LABEL_MAPPING['CORUM'] == 'in complex'

    print("✓ KG utils tests passed")


def test_formatter():
    """Test formatter with new edge types."""
    from src.generator.formatter import format_subgraph
    import networkx as nx

    # Create a test subgraph
    subgraph = nx.DiGraph()
    subgraph.add_node('TP53', node_label='gene', name='Tumor protein p53')
    subgraph.add_node('OMIM:123456', node_label='disease', name='Test Disease')
    subgraph.add_node('GTEx:Liver', node_label='tissue', name='Liver')
    subgraph.add_node('CORUM:1', node_label='complex', name='Test Complex')

    subgraph.add_edge('TP53', 'OMIM:123456', edge_type='OMIM', inheritance='AD')
    subgraph.add_edge('TP53', 'GTEx:Liver', edge_type='GTEx', tpm=100.5, tissue_specificity=0.8)
    subgraph.add_edge('TP53', 'CORUM:1', edge_type='CORUM', complex_name='Test Complex', subunit_count=5)

    result = format_subgraph(subgraph, 'TP53')

    assert 'disease_info' in result
    assert 'tissue_info' in result
    assert 'complex_info' in result

    print("✓ Formatter tests passed")


def test_adapters():
    """Test adapter imports."""
    from kg_builder.adapters import (
        HGNCAdapter, STRINGAdapter, TRRUSTAdapter,
        GOAdapter, HPOAdapter, CellMarkerAdapter, ReactomeAdapter,
        OMIMAdapter, GTExAdapter, CORUMAdapter
    )

    print("✓ Adapter imports passed")


def main():
    print("=" * 50)
    print("SIGR Framework Update Validation")
    print("=" * 50)

    try:
        test_strategy_config()
        test_prompts()
        test_kg_utils()
        test_formatter()
        test_adapters()

        print("=" * 50)
        print("All validation tests passed!")
        print("=" * 50)
        return 0

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
