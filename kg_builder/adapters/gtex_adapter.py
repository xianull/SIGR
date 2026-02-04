"""
GTEx Adapter for BioCypher

Loads tissue-specific gene expression data from GTEx.

Data source: https://gtexportal.org/
Output: Tissue nodes + EXPRESSED_IN_TISSUE edges
"""

import logging
from pathlib import Path
from typing import Generator, Set, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class GTExAdapter:
    """
    BioCypher adapter for GTEx tissue expression data.

    Node type: biolink:GrossAnatomicalStructure (Tissue)
    Edge type: biolink:GeneToExpressionSiteAssociation
    Relationship label: EXPRESSED_IN_TISSUE

    Expected data file: GTEx_Analysis_*_TPM.gct or median expression file
    """

    # Standard GTEx tissue names
    TISSUE_MAPPING = {
        'Adipose_Subcutaneous': 'Adipose - Subcutaneous',
        'Adipose_Visceral_Omentum': 'Adipose - Visceral (Omentum)',
        'Adrenal_Gland': 'Adrenal Gland',
        'Artery_Aorta': 'Artery - Aorta',
        'Artery_Coronary': 'Artery - Coronary',
        'Artery_Tibial': 'Artery - Tibial',
        'Bladder': 'Bladder',
        'Brain_Amygdala': 'Brain - Amygdala',
        'Brain_Anterior_cingulate_cortex_BA24': 'Brain - Anterior cingulate cortex (BA24)',
        'Brain_Caudate_basal_ganglia': 'Brain - Caudate (basal ganglia)',
        'Brain_Cerebellar_Hemisphere': 'Brain - Cerebellar Hemisphere',
        'Brain_Cerebellum': 'Brain - Cerebellum',
        'Brain_Cortex': 'Brain - Cortex',
        'Brain_Frontal_Cortex_BA9': 'Brain - Frontal Cortex (BA9)',
        'Brain_Hippocampus': 'Brain - Hippocampus',
        'Brain_Hypothalamus': 'Brain - Hypothalamus',
        'Brain_Nucleus_accumbens_basal_ganglia': 'Brain - Nucleus accumbens (basal ganglia)',
        'Brain_Putamen_basal_ganglia': 'Brain - Putamen (basal ganglia)',
        'Brain_Spinal_cord_cervical_c-1': 'Brain - Spinal cord (cervical c-1)',
        'Brain_Substantia_nigra': 'Brain - Substantia nigra',
        'Breast_Mammary_Tissue': 'Breast - Mammary Tissue',
        'Cells_Cultured_fibroblasts': 'Cells - Cultured fibroblasts',
        'Cells_EBV-transformed_lymphocytes': 'Cells - EBV-transformed lymphocytes',
        'Colon_Sigmoid': 'Colon - Sigmoid',
        'Colon_Transverse': 'Colon - Transverse',
        'Esophagus_Gastroesophageal_Junction': 'Esophagus - Gastroesophageal Junction',
        'Esophagus_Mucosa': 'Esophagus - Mucosa',
        'Esophagus_Muscularis': 'Esophagus - Muscularis',
        'Heart_Atrial_Appendage': 'Heart - Atrial Appendage',
        'Heart_Left_Ventricle': 'Heart - Left Ventricle',
        'Kidney_Cortex': 'Kidney - Cortex',
        'Kidney_Medulla': 'Kidney - Medulla',
        'Liver': 'Liver',
        'Lung': 'Lung',
        'Minor_Salivary_Gland': 'Minor Salivary Gland',
        'Muscle_Skeletal': 'Muscle - Skeletal',
        'Nerve_Tibial': 'Nerve - Tibial',
        'Ovary': 'Ovary',
        'Pancreas': 'Pancreas',
        'Pituitary': 'Pituitary',
        'Prostate': 'Prostate',
        'Skin_Not_Sun_Exposed_Suprapubic': 'Skin - Not Sun Exposed (Suprapubic)',
        'Skin_Sun_Exposed_Lower_leg': 'Skin - Sun Exposed (Lower leg)',
        'Small_Intestine_Terminal_Ileum': 'Small Intestine - Terminal Ileum',
        'Spleen': 'Spleen',
        'Stomach': 'Stomach',
        'Testis': 'Testis',
        'Thyroid': 'Thyroid',
        'Uterus': 'Uterus',
        'Vagina': 'Vagina',
        'Whole_Blood': 'Whole Blood',
    }

    def __init__(
        self,
        data_path: str,
        tpm_threshold: float = 1.0,
        tissue_specificity_threshold: float = 0.0
    ):
        """
        Initialize GTEx adapter.

        Args:
            data_path: Path to GTEx expression file (median TPM per tissue)
            tpm_threshold: Minimum TPM to consider gene as expressed
            tissue_specificity_threshold: Minimum tissue specificity index (0-1)
        """
        self.data_path = Path(data_path)
        self.tpm_threshold = tpm_threshold
        self.tissue_specificity_threshold = tissue_specificity_threshold

        self._tissues: Dict[str, Dict] = {}
        self._expressions: list = []
        self._loaded = False

    def _load_data(self) -> None:
        """Load GTEx expression data."""
        if self._loaded:
            return

        logger.info(f"Loading GTEx data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"GTEx file not found: {self.data_path}")

        # Detect file format
        if self.data_path.suffix == '.gct':
            self._load_gct()
        else:
            self._load_tsv()

        self._loaded = True
        logger.info(
            f"Loaded {len(self._tissues)} tissues, "
            f"{len(self._expressions)} expression entries"
        )

    def _load_gct(self) -> None:
        """Load from GCT format."""
        # GCT format has 2 header lines, then data
        df = pd.read_csv(self.data_path, sep='\t', skiprows=2)

        # First two columns are Name (gene ID) and Description (gene name)
        gene_col = df.columns[0]
        name_col = df.columns[1]
        tissue_cols = df.columns[2:]

        self._process_expression_matrix(df, gene_col, name_col, tissue_cols)

    def _load_tsv(self) -> None:
        """Load from TSV format."""
        df = pd.read_csv(self.data_path, sep='\t')

        # Try to identify columns
        gene_col = None
        name_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'gene_id' in col_lower or 'ensembl' in col_lower:
                gene_col = col
            elif 'gene_name' in col_lower or 'symbol' in col_lower:
                name_col = col

        if gene_col is None:
            gene_col = df.columns[0]
        if name_col is None:
            name_col = df.columns[1] if len(df.columns) > 1 else gene_col

        # Tissue columns are the rest
        exclude_cols = {gene_col, name_col, 'Description', 'description'}
        tissue_cols = [c for c in df.columns if c not in exclude_cols and not c.lower().startswith('gene')]

        self._process_expression_matrix(df, gene_col, name_col, tissue_cols)

    def _process_expression_matrix(
        self,
        df: pd.DataFrame,
        gene_col: str,
        name_col: str,
        tissue_cols: list
    ) -> None:
        """Process expression matrix to extract edges."""
        # Register tissues
        for tissue_col in tissue_cols:
            tissue_id = f"GTEx:{tissue_col.replace(' ', '_')}"
            tissue_name = self.TISSUE_MAPPING.get(tissue_col, tissue_col.replace('_', ' '))

            self._tissues[tissue_id] = {
                'tissue_id': tissue_id,
                'name': tissue_name,
                'source_col': tissue_col,
            }

        # Process genes
        for _, row in df.iterrows():
            gene_id = row.get(gene_col)
            gene_name = row.get(name_col)

            if pd.isna(gene_id):
                continue

            # Extract gene symbol from ENSEMBL ID if needed
            gene_symbol = str(gene_name).strip().upper() if pd.notna(gene_name) else None

            # If gene_id is ENSEMBL format and we have a symbol, use symbol
            if gene_symbol and not str(gene_id).startswith('ENSG'):
                gene_symbol = str(gene_id).strip().upper()

            if not gene_symbol:
                continue

            # Calculate expression values and tissue specificity
            expression_values = {}
            for tissue_col in tissue_cols:
                tpm = row.get(tissue_col)
                if pd.notna(tpm):
                    try:
                        expression_values[tissue_col] = float(tpm)
                    except (ValueError, TypeError):
                        pass

            if not expression_values:
                continue

            # Calculate tissue specificity (tau)
            max_expr = max(expression_values.values())
            if max_expr > 0:
                tau = self._calculate_tau(list(expression_values.values()))
            else:
                tau = 0.0

            # Create expression edges for tissues above threshold
            for tissue_col, tpm in expression_values.items():
                if tpm >= self.tpm_threshold:
                    tissue_id = f"GTEx:{tissue_col.replace(' ', '_')}"

                    # Calculate relative expression
                    relative_expr = tpm / max_expr if max_expr > 0 else 0.0

                    self._expressions.append({
                        'gene': gene_symbol,
                        'tissue_id': tissue_id,
                        'tpm': tpm,
                        'relative_expression': relative_expr,
                        'tissue_specificity': tau,
                    })

    def _calculate_tau(self, expression_values: list) -> float:
        """
        Calculate tissue specificity index (tau).

        Tau ranges from 0 (ubiquitous) to 1 (tissue-specific).
        """
        if not expression_values or max(expression_values) == 0:
            return 0.0

        max_expr = max(expression_values)
        n = len(expression_values)

        if n <= 1:
            return 0.0

        # Tau = sum(1 - x_i/max_x) / (n - 1)
        tau_sum = sum(1 - (x / max_expr) for x in expression_values)
        tau = tau_sum / (n - 1)

        return min(1.0, max(0.0, tau))

    def get_nodes(self) -> Generator[tuple, None, None]:
        """
        Yield tissue nodes in BioCypher format.

        Yields:
            Tuple of (node_id, node_label, properties)
        """
        self._load_data()

        for tissue_id, info in self._tissues.items():
            properties = {
                'tissue_id': tissue_id,
                'name': info['name'],
                'source_database': 'GTEx',
            }

            yield (tissue_id, 'Tissue', properties)

    def get_edges(self) -> Generator[tuple, None, None]:
        """
        Yield gene-tissue expression edges in BioCypher format.

        Yields:
            Tuple of (source_id, target_id, relationship_label, properties)
        """
        self._load_data()

        # Filter by tissue specificity threshold
        filtered = [
            e for e in self._expressions
            if e['tissue_specificity'] >= self.tissue_specificity_threshold
        ]

        # Deduplicate (keep highest TPM per gene-tissue pair)
        best_expr: Dict[tuple, dict] = {}
        for expr in filtered:
            key = (expr['gene'], expr['tissue_id'])
            if key not in best_expr or expr['tpm'] > best_expr[key]['tpm']:
                best_expr[key] = expr

        for (gene, tissue_id), expr in best_expr.items():
            properties = {
                'source_database': 'GTEx',
                'tpm': expr['tpm'],
                'relative_expression': expr['relative_expression'],
                'tissue_specificity': expr['tissue_specificity'],
            }

            yield (gene, tissue_id, 'EXPRESSED_IN_TISSUE', properties)

    def get_node_count(self) -> int:
        """Get number of tissues."""
        self._load_data()
        return len(self._tissues)

    def get_edge_count(self) -> int:
        """Get number of expression entries."""
        self._load_data()
        return len(self._expressions)
