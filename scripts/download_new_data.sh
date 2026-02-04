#!/bin/bash
# SIGR数据下载脚本
# 下载新增的数据源：OMIM、GTEx、CORUM

set -e

DATA_DIR="${1:-/Users/xianull/Documents/Project/SIGR/data/raw}"
cd "$DATA_DIR"

echo "=========================================="
echo "SIGR 新数据源下载脚本"
echo "目标目录: $DATA_DIR"
echo "=========================================="

# 1. CORUM - 蛋白复合物数据
echo ""
echo "[1/3] 下载 CORUM 蛋白复合物数据..."
if [ ! -f "corum_coreComplexes.txt" ]; then
    # 尝试主站点
    curl -L -o corum_coreComplexes.txt \
        "http://mips.helmholtz-muenchen.de/corum/download/coreComplexes.txt" \
        --retry 3 --retry-delay 5 || \
    # 备用：尝试压缩版
    (curl -L -o corum_allComplexes.txt.zip \
        "https://mips.helmholtz-muenchen.de/corum/download/allComplexes.txt.zip" \
        --retry 3 && unzip -o corum_allComplexes.txt.zip && rm corum_allComplexes.txt.zip) || \
    echo "警告: CORUM下载失败，请手动下载"

    if [ -f "corum_coreComplexes.txt" ]; then
        echo "✓ CORUM 下载完成: corum_coreComplexes.txt"
    fi
else
    echo "✓ CORUM 已存在: corum_coreComplexes.txt"
fi

# 2. GTEx - 组织表达数据
echo ""
echo "[2/3] 下载 GTEx 组织表达数据..."
if [ ! -f "GTEx_gene_median_tpm.gct.gz" ] && [ ! -f "GTEx_gene_median_tpm.gct" ]; then
    # GTEx V8 中位TPM数据
    curl -L -o GTEx_gene_median_tpm.gct.gz \
        "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz" \
        --retry 3 --retry-delay 10 || \
    echo "警告: GTEx下载失败，请手动从 https://gtexportal.org/ 下载"

    if [ -f "GTEx_gene_median_tpm.gct.gz" ]; then
        echo "✓ GTEx 下载完成: GTEx_gene_median_tpm.gct.gz"
        echo "  解压中..."
        gunzip -k GTEx_gene_median_tpm.gct.gz 2>/dev/null || true
    fi
else
    echo "✓ GTEx 已存在"
fi

# 3. OMIM - 疾病-基因关联
echo ""
echo "[3/3] OMIM 数据说明..."
echo "  OMIM 需要注册才能下载数据。"
echo "  请访问: https://omim.org/downloads/"
echo "  注册后下载以下文件到 $DATA_DIR:"
echo "    - morbidmap.txt (推荐) 或 genemap2.txt"
echo ""
if [ -f "morbidmap.txt" ] || [ -f "genemap2.txt" ]; then
    echo "✓ OMIM 数据已存在"
else
    echo "⚠ OMIM 数据缺失，请手动下载"
fi

# 打印摘要
echo ""
echo "=========================================="
echo "下载摘要"
echo "=========================================="
echo "CORUM: $([ -f 'corum_coreComplexes.txt' ] && echo '✓ 存在' || echo '✗ 缺失')"
echo "GTEx:  $([ -f 'GTEx_gene_median_tpm.gct.gz' ] || [ -f 'GTEx_gene_median_tpm.gct' ] && echo '✓ 存在' || echo '✗ 缺失')"
echo "OMIM:  $([ -f 'morbidmap.txt' ] || [ -f 'genemap2.txt' ] && echo '✓ 存在' || echo '✗ 缺失 (需手动下载)')"
echo ""
echo "手动下载链接:"
echo "- CORUM: https://mips.helmholtz-muenchen.de/corum/#download"
echo "- GTEx:  https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression"
echo "- OMIM:  https://omim.org/downloads/ (需注册)"
echo "=========================================="
