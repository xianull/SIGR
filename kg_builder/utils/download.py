"""
Data Download Utility

Handles downloading raw data files from various biological databases.
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
from urllib.parse import urlparse
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Data source URLs and metadata
DATA_SOURCES = {
    # HGNC gene nomenclature (using genenames.org direct link)
    "hgnc": {
        "url": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
        "filename": "hgnc_complete_set.txt",
        "description": "HGNC complete gene set",
    },
    # TRRUST TF-target database
    "trrust": {
        "url": "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
        "filename": "trrust_rawdata.human.tsv",
        "description": "TRRUST v2 human TF-target regulations",
    },
    # STRING PPI (human, full network) - using versioned URL
    "string_links": {
        "url": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        "filename": "9606.protein.links.v12.0.txt.gz",
        "description": "STRING v12 human protein links",
    },
    "string_info": {
        "url": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
        "filename": "9606.protein.info.v12.0.txt.gz",
        "description": "STRING v12 human protein info",
    },
    # STRING protein aliases (for ID mapping)
    "string_aliases": {
        "url": "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
        "filename": "9606.protein.aliases.v12.0.txt.gz",
        "description": "STRING v12 human protein aliases (for gene symbol mapping)",
    },
    # Gene Ontology annotations
    "go_annotations": {
        "url": "http://geneontology.org/gene-associations/goa_human.gaf.gz",
        "filename": "goa_human.gaf.gz",
        "description": "GO annotations for human genes",
    },
    # HPO gene-phenotype associations
    "hpo_genes": {
        "url": "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2024-04-26/genes_to_phenotype.txt",
        "filename": "genes_to_phenotype.txt",
        "description": "HPO gene to phenotype associations",
    },
    # DisGeNET gene-disease associations
    "disgenet": {
        "url": "https://www.disgenet.org/static/disgenet_ap1/files/downloads/curated_gene_disease_associations.tsv.gz",
        "filename": "curated_gene_disease_associations.tsv.gz",
        "description": "DisGeNET curated gene-disease associations",
        "note": "May require registration for full dataset",
    },
    # PanglaoDB markers - direct TSV link
    "panglaodb": {
        "url": "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz",
        "filename": "PanglaoDB_markers.tsv.gz",
        "description": "PanglaoDB cell type markers",
    },
}

# Manual download instructions for restricted datasets
MANUAL_DOWNLOADS = {
    "cellmarker": {
        "url": "http://biocc.hrbmu.edu.cn/CellMarker/download.jsp",
        "filename": "CellMarker_Human.xlsx",
        "instructions": "Download 'Human Cell Marker' Excel file from the website",
    },
    "depmap": {
        "url": "https://depmap.org/portal/download/all/",
        "filename": "CRISPRGeneEffect.csv",
        "instructions": "Download 'CRISPR Gene Effect' from DepMap portal (requires registration)",
    },
    "epigenomics_h3k4me3": {
        "url": "https://egg2.wustl.edu/roadmap/",
        "filename": "H3K4me3_genes.txt",
        "instructions": "Process H3K4me3 ChIP-seq peak files from Roadmap Epigenomics",
    },
    "epigenomics_h3k27me3": {
        "url": "https://egg2.wustl.edu/roadmap/",
        "filename": "H3K27me3_genes.txt",
        "instructions": "Process H3K27me3 ChIP-seq peak files from Roadmap Epigenomics",
    },
}


class DataDownloader:
    """
    Utility class for downloading biological database files.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize downloader.

        Args:
            data_dir: Directory to save downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(
        self,
        url: str,
        filename: str,
        force: bool = False,
        chunk_size: int = 8192
    ) -> Path:
        """
        Download a file from URL.

        Args:
            url: URL to download from
            filename: Local filename to save as
            force: If True, re-download even if file exists
            chunk_size: Download chunk size

        Returns:
            Path to downloaded file
        """
        filepath = self.data_dir / filename

        if filepath.exists() and not force:
            logger.info(f"File already exists: {filepath}")
            return filepath

        logger.info(f"Downloading {url} to {filepath}")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Downloaded: {filepath}")
            return filepath

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            raise

    def download_source(self, source_name: str, force: bool = False) -> Optional[Path]:
        """
        Download a predefined data source.

        Args:
            source_name: Name of data source (from DATA_SOURCES)
            force: If True, re-download even if file exists

        Returns:
            Path to downloaded file or None if manual download required
        """
        if source_name in DATA_SOURCES:
            source = DATA_SOURCES[source_name]
            return self.download_file(
                url=source["url"],
                filename=source["filename"],
                force=force
            )
        elif source_name in MANUAL_DOWNLOADS:
            source = MANUAL_DOWNLOADS[source_name]
            logger.warning(
                f"Manual download required for '{source_name}':\n"
                f"  URL: {source['url']}\n"
                f"  Save as: {self.data_dir / source['filename']}\n"
                f"  Instructions: {source['instructions']}"
            )
            return None
        else:
            raise ValueError(f"Unknown data source: {source_name}")

    def download_all(self, force: bool = False) -> Dict[str, Optional[Path]]:
        """
        Download all available data sources.

        Args:
            force: If True, re-download all files

        Returns:
            Dictionary mapping source names to file paths
        """
        results = {}

        for source_name in DATA_SOURCES:
            try:
                results[source_name] = self.download_source(source_name, force)
            except Exception as e:
                logger.error(f"Failed to download {source_name}: {e}")
                results[source_name] = None

        return results

    def check_files(self) -> Dict[str, bool]:
        """
        Check which data files exist.

        Returns:
            Dictionary mapping source names to existence status
        """
        status = {}

        for source_name, source in DATA_SOURCES.items():
            filepath = self.data_dir / source["filename"]
            status[source_name] = filepath.exists()

        for source_name, source in MANUAL_DOWNLOADS.items():
            filepath = self.data_dir / source["filename"]
            status[source_name] = filepath.exists()

        return status

    def get_file_path(self, source_name: str) -> Path:
        """
        Get the expected file path for a data source.

        Args:
            source_name: Name of data source

        Returns:
            Path where the file should be located
        """
        if source_name in DATA_SOURCES:
            return self.data_dir / DATA_SOURCES[source_name]["filename"]
        elif source_name in MANUAL_DOWNLOADS:
            return self.data_dir / MANUAL_DOWNLOADS[source_name]["filename"]
        else:
            raise ValueError(f"Unknown data source: {source_name}")

    @staticmethod
    def list_sources() -> None:
        """Print available data sources."""
        print("\n=== Automatic Downloads ===")
        for name, source in DATA_SOURCES.items():
            print(f"\n{name}:")
            print(f"  Description: {source['description']}")
            print(f"  URL: {source['url']}")
            print(f"  Filename: {source['filename']}")

        print("\n=== Manual Downloads Required ===")
        for name, source in MANUAL_DOWNLOADS.items():
            print(f"\n{name}:")
            print(f"  URL: {source['url']}")
            print(f"  Filename: {source['filename']}")
            print(f"  Instructions: {source['instructions']}")


def main():
    """CLI interface for data download."""
    import argparse

    parser = argparse.ArgumentParser(description="Download biological database files")
    parser.add_argument("--list", action="store_true", help="List available data sources")
    parser.add_argument("--download", nargs="*", help="Download specific sources (or all if none specified)")
    parser.add_argument("--check", action="store_true", help="Check which files exist")
    parser.add_argument("--data-dir", default="data/raw", help="Data directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    if args.list:
        DataDownloader.list_sources()
        return

    downloader = DataDownloader(args.data_dir)

    if args.check:
        status = downloader.check_files()
        print("\n=== File Status ===")
        for name, exists in status.items():
            status_str = "✓ EXISTS" if exists else "✗ MISSING"
            print(f"  {name}: {status_str}")
        return

    if args.download is not None:
        if len(args.download) == 0:
            # Download all
            results = downloader.download_all(args.force)
        else:
            # Download specific sources
            results = {}
            for source in args.download:
                results[source] = downloader.download_source(source, args.force)

        print("\n=== Download Results ===")
        for name, path in results.items():
            if path:
                print(f"  {name}: {path}")
            else:
                print(f"  {name}: MANUAL DOWNLOAD REQUIRED")


if __name__ == "__main__":
    main()
