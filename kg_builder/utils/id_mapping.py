"""
Gene ID Mapping Utility

Provides conversion between different gene identifier systems:
- HGNC Symbol (primary)
- Ensembl Gene ID
- NCBI Gene ID (Entrez)
- UniProt ID

Uses mygene.info API for conversions.
"""

import logging
from typing import Optional, Dict, List, Set, Union
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)


class GeneIDMapper:
    """
    Gene ID conversion utility using HGNC and mygene.info.

    Primary ID: HGNC Symbol
    Supports conversion to/from: Ensembl, NCBI/Entrez, UniProt
    """

    def __init__(self, hgnc_path: Optional[str] = None):
        """
        Initialize the mapper.

        Args:
            hgnc_path: Path to HGNC complete set TSV file.
                      If None, will use mygene.info API only.
        """
        self._hgnc_df = None
        self._symbol_to_ensembl: Dict[str, str] = {}
        self._symbol_to_entrez: Dict[str, str] = {}
        self._symbol_to_uniprot: Dict[str, List[str]] = {}
        self._ensembl_to_symbol: Dict[str, str] = {}
        self._entrez_to_symbol: Dict[str, str] = {}
        self._alias_to_symbol: Dict[str, str] = {}

        if hgnc_path:
            self._load_hgnc(hgnc_path)

    def _load_hgnc(self, path: str) -> None:
        """Load HGNC complete set for ID mapping."""
        logger.info(f"Loading HGNC data from {path}")

        try:
            self._hgnc_df = pd.read_csv(path, sep='\t', low_memory=False)

            # Build mapping dictionaries
            for _, row in self._hgnc_df.iterrows():
                symbol = row.get('symbol')
                if pd.isna(symbol):
                    continue

                # Ensembl mapping
                ensembl = row.get('ensembl_gene_id')
                if pd.notna(ensembl):
                    self._symbol_to_ensembl[symbol] = ensembl
                    self._ensembl_to_symbol[ensembl] = symbol

                # Entrez/NCBI mapping
                entrez = row.get('entrez_id')
                if pd.notna(entrez):
                    entrez_str = str(int(entrez))
                    self._symbol_to_entrez[symbol] = entrez_str
                    self._entrez_to_symbol[entrez_str] = symbol

                # UniProt mapping (can be multiple)
                uniprot = row.get('uniprot_ids')
                if pd.notna(uniprot):
                    uniprot_list = str(uniprot).split('|')
                    self._symbol_to_uniprot[symbol] = uniprot_list

                # Alias mapping (previous symbols and aliases)
                prev_symbols = row.get('prev_symbol')
                if pd.notna(prev_symbols):
                    for alias in str(prev_symbols).split('|'):
                        alias = alias.strip()
                        if alias:
                            self._alias_to_symbol[alias] = symbol

                alias_symbols = row.get('alias_symbol')
                if pd.notna(alias_symbols):
                    for alias in str(alias_symbols).split('|'):
                        alias = alias.strip()
                        if alias:
                            self._alias_to_symbol[alias] = symbol

            logger.info(
                f"Loaded {len(self._symbol_to_ensembl)} symbol-Ensembl mappings, "
                f"{len(self._symbol_to_entrez)} symbol-Entrez mappings, "
                f"{len(self._alias_to_symbol)} alias mappings"
            )

        except Exception as e:
            logger.error(f"Failed to load HGNC data: {e}")
            raise

    def symbol_to_ensembl(self, symbol: str) -> Optional[str]:
        """Convert HGNC symbol to Ensembl gene ID."""
        return self._symbol_to_ensembl.get(symbol)

    def symbol_to_entrez(self, symbol: str) -> Optional[str]:
        """Convert HGNC symbol to NCBI/Entrez gene ID."""
        return self._symbol_to_entrez.get(symbol)

    def symbol_to_uniprot(self, symbol: str) -> List[str]:
        """Convert HGNC symbol to UniProt ID(s)."""
        return self._symbol_to_uniprot.get(symbol, [])

    def ensembl_to_symbol(self, ensembl_id: str) -> Optional[str]:
        """Convert Ensembl gene ID to HGNC symbol."""
        return self._ensembl_to_symbol.get(ensembl_id)

    def entrez_to_symbol(self, entrez_id: str) -> Optional[str]:
        """Convert NCBI/Entrez gene ID to HGNC symbol."""
        return self._entrez_to_symbol.get(str(entrez_id))

    def normalize_symbol(self, query: str) -> Optional[str]:
        """
        Normalize a gene identifier to official HGNC symbol.

        Handles:
        - Already official symbols
        - Previous symbols
        - Aliases

        Args:
            query: Gene identifier to normalize

        Returns:
            Official HGNC symbol or None if not found
        """
        query = query.strip().upper()

        # Check if already official symbol
        if query in self._symbol_to_ensembl or query in self._symbol_to_entrez:
            return query

        # Check aliases
        if query in self._alias_to_symbol:
            return self._alias_to_symbol[query]

        return None

    def batch_convert(
        self,
        ids: List[str],
        from_type: str = "symbol",
        to_type: str = "ensembl"
    ) -> Dict[str, Optional[str]]:
        """
        Batch convert gene IDs.

        Args:
            ids: List of gene identifiers
            from_type: Source ID type (symbol, ensembl, entrez)
            to_type: Target ID type (symbol, ensembl, entrez)

        Returns:
            Dictionary mapping input IDs to converted IDs
        """
        results = {}

        for gene_id in ids:
            if from_type == "symbol" and to_type == "ensembl":
                results[gene_id] = self.symbol_to_ensembl(gene_id)
            elif from_type == "symbol" and to_type == "entrez":
                results[gene_id] = self.symbol_to_entrez(gene_id)
            elif from_type == "ensembl" and to_type == "symbol":
                results[gene_id] = self.ensembl_to_symbol(gene_id)
            elif from_type == "entrez" and to_type == "symbol":
                results[gene_id] = self.entrez_to_symbol(gene_id)
            else:
                results[gene_id] = None

        return results

    def get_all_symbols(self) -> Set[str]:
        """Get set of all known HGNC symbols."""
        return set(self._symbol_to_ensembl.keys()) | set(self._symbol_to_entrez.keys())

    def get_gene_info(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive gene information for a symbol.

        Returns:
            Dictionary with gene information or None
        """
        if self._hgnc_df is None:
            return None

        rows = self._hgnc_df[self._hgnc_df['symbol'] == symbol]
        if len(rows) == 0:
            return None

        row = rows.iloc[0]
        return {
            'symbol': row.get('symbol'),
            'name': row.get('name'),
            'hgnc_id': row.get('hgnc_id'),
            'ensembl_gene_id': row.get('ensembl_gene_id'),
            'entrez_id': row.get('entrez_id'),
            'chromosome': row.get('location'),
            'gene_type': row.get('locus_type'),
            'gene_group': row.get('gene_group'),
            'uniprot_ids': row.get('uniprot_ids'),
        }


# Global instance for convenience
_mapper_instance: Optional[GeneIDMapper] = None


def get_mapper(hgnc_path: Optional[str] = None) -> GeneIDMapper:
    """
    Get or create a global GeneIDMapper instance.

    Args:
        hgnc_path: Path to HGNC data (only used on first call)

    Returns:
        GeneIDMapper instance
    """
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = GeneIDMapper(hgnc_path)
    return _mapper_instance
