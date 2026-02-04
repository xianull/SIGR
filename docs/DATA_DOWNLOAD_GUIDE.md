# SIGR 新数据源下载指南

由于网络问题，请手动下载以下数据文件到 `data/raw/` 目录。

## 当前已有数据 ✅

| 数据源 | 文件 | 状态 |
|--------|------|------|
| STRING | 9606.protein.links.v12.0.txt.gz | ✅ 存在 |
| HGNC | hgnc_complete_set.txt | ✅ 存在 |
| GO | goa_human.gaf.gz | ✅ 存在 |
| HPO | genes_to_phenotype.txt | ✅ 存在 |
| CellMarker | Cell_marker_Human.xlsx | ✅ 存在 |
| TRRUST | trrust_rawdata.human.tsv | ✅ 存在 |
| Reactome | Reactome/*.txt | ✅ 存在 |

## 需要下载的新数据 ❌

### 1. CORUM (蛋白复合物数据)

**下载链接**: https://mips.helmholtz-muenchen.de/corum/#download

**操作步骤**:
1. 访问上述链接
2. 下载 `coreComplexes.txt` 或 `allComplexes.txt`
3. 重命名为 `corum_coreComplexes.txt`
4. 放入 `data/raw/` 目录

**预期文件**: `data/raw/corum_coreComplexes.txt`

---

### 2. GTEx (组织表达数据)

**下载链接**: https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression

**操作步骤**:
1. 访问上述链接
2. 下载 `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz`
3. 放入 `data/raw/` 目录
4. 可选：解压 `gunzip GTEx_*.gct.gz`

**预期文件**: `data/raw/GTEx_gene_median_tpm.gct.gz` 或 `.gct`

---

### 3. OMIM (疾病-基因关联) - 需要注册

**下载链接**: https://omim.org/downloads/

**操作步骤**:
1. 访问上述链接
2. **注册账号** (需要机构邮箱)
3. 申请数据访问权限
4. 下载 `morbidmap.txt` 或 `genemap2.txt`
5. 放入 `data/raw/` 目录

**预期文件**: `data/raw/morbidmap.txt` 或 `data/raw/genemap2.txt`

---

## 验证下载

下载完成后，运行以下命令验证：

```bash
ls -la data/raw/corum*.txt data/raw/GTEx*.gct* data/raw/morbidmap.txt data/raw/genemap*.txt 2>/dev/null
```

## 运行知识图谱构建

数据准备好后，运行：

```bash
# 仅使用已有数据（不含新数据源）
python kg_builder/build_kg.py

# 包含CORUM（如果已下载）
python kg_builder/build_kg.py --include-corum

# 包含所有新数据源（如果都已下载）
python kg_builder/build_kg.py --include-omim --include-gtex --include-corum
```

## 注意事项

1. **OMIM** 数据需要注册和机构邮箱，可能需要几天审批
2. **GTEx** 文件较大（约1.5GB），下载可能需要一些时间
3. 如果某个数据源暂时不可用，可以先跳过，不影响其他功能
