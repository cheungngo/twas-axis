# Psychiatric Disease Proximity Pipeline - TWAS-AXIS

A computational pipeline that projects a new psychiatric phenotype into a three-axis space defined by **Major Depressive Disorder (MDD)**, **Bipolar Disorder (BIP)**, and **Obsessive-Compulsive Disorder (OCD)** using curated neurobiological gene-set signatures derived from brain-region [S-PrediXcan](https://github.com/hakyimlab/MetaXcan) (TWAS) results.

---

## Table of Contents

- [Overview](#overview)
- [Curated Pathway Categories](#curated-pathway-categories)
- [Installation](#installation)
- [Input Format](#input-format)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Python API](#python-api)
- [Output](#output)
- [Methods](#methods)
  - [Meta-Analysis Across Brain Regions](#meta-analysis-across-brain-regions)
  - [Pathway Scoring](#pathway-scoring)
  - [Proximity Metrics](#proximity-metrics)
  - [Ternary Projection](#ternary-projection)
- [Example](#example)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## Overview

Psychiatric disorders share substantial genetic architecture, yet their neurobiological overlap at the level of specific pathways remains poorly characterised. This pipeline quantifies how closely a new psychiatric phenotype resembles three canonical disorders — MDD, BIP, and OCD — across curated neurobiological gene sets spanning neurotransmission, synaptic plasticity, glial biology, neuroimmunity, and cellular ageing.

The pipeline proceeds in five stages:

1. **Load** S-PrediXcan association z-scores from multiple brain-region tissue models for each disorder.
2. **Meta-analyse** z-scores across brain regions using Stouffer's method to obtain a single gene-level summary statistic per curated gene.
3. **Score** 23 curated neurobiological pathways (collapsible to 13 broad categories) using directional enrichment (Stouffer Z) and dysregulation magnitude (mean |Z|).
4. **Compute** proximity between the new disease and each reference axis using Pearson correlation, Spearman correlation, cosine similarity, and Euclidean distance — at the pathway level and gene level.
5. **Visualise** results as heatmaps, radar plots, PCA biplots, ternary diagrams, and grouped bar charts.

---

## Curated Pathway Categories

The pipeline evaluates 13 broad neurobiological categories. The monoamine neurotransmission category expands into 11 sub-categories for fine-grained analysis, yielding 23 detailed pathways in total.

| Category | Detailed Pathways | Gene Count | Source |
|----------|-------------------|------------|--------|
| Monoamine neurotransmission | Dopamine receptors, Dopamine synthesis/metabolism, Dopamine transport/signalling, Serotonin receptors, Serotonin synthesis/metabolism, Norepinephrine receptors, Norepinephrine synthesis/transport, Histamine system, Trace amine receptors, Downstream signalling, Vesicular machinery | 111 | Literature curation |
| Synaptic plasticity | GOBP Regulation of Synaptic Plasticity | 214 | GO:0048167 via MSigDB |
| Long-term potentiation | KEGG LTP | 70 | hsa04720 via MSigDB |
| Glutamatergic transmission | GOBP Glutamatergic Synaptic Transmission | 109 | GO:0035249 via MSigDB |
| Synapse pruning | GOBP Synapse Pruning | 18 | GO:0098883 via MSigDB |
| Complement cascade | Reactome Complement Cascade | 113 | R-HSA-166658 via MSigDB |
| Cellular senescence | Reactome Cellular Senescence | 189 | R-HSA-2559583 via MSigDB |
| Telomere maintenance | Reactome Telomere Maintenance | 107 | R-HSA-157579 via MSigDB |
| Astrocyte markers | Adult astrocyte signature | 40 | Zhang et al., *Neuron* 2016 |
| Oligodendrocyte markers | Adult oligodendrocyte signature | 35 | Zhang et al., *Neuron* 2016 |
| Microglia markers | Adult microglia signature | 25 | Zhang et al., *Neuron* 2016 |
| HLA complex | All HLA genes | 44 | HGNC gene group 588 |

---

## Installation

### From source

```bash
git clone https://github.com/cheungngo/twas-axis.git
cd psychiatric-proximity
pip install -r requirements.txt
```

### As an installable package

```bash
git clone https://github.com/cheungngo/twas-axis.git
cd psychiatric-proximity
pip install .
```

When installed as a package, the command `psychiatric-proximity` becomes available system-wide.

### Requirements

- Python ≥ 3.8
- numpy ≥ 1.21
- pandas ≥ 1.3
- scipy ≥ 1.7
- matplotlib ≥ 3.5

No GPU or special hardware is required. The pipeline runs in under a minute on typical input sizes.

---

## Input Format

Each disorder requires a **folder** containing one or more S-PrediXcan result files — typically one file per brain-region tissue model. Files may be in CSV (comma-separated) or TSV (tab-separated) format, including `.csv`, `.tsv`, `.txt`, and `.dat` extensions. The pipeline auto-detects the delimiter.

### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `gene_name` | string | HGNC gene symbol (e.g. `DRD2`, `HTR2A`, `GRIN1`) |
| `zscore` | float | S-PrediXcan association z-score |

### Optional columns

| Column | Type | Description |
|--------|------|-------------|
| `pvalue` | float | When present, duplicate gene entries are resolved by keeping the row with the smallest p-value |

### Example file structure

```
data/
├── mdd/
│   ├── Brain_Amygdala.csv
│   ├── Brain_Anterior_cingulate_cortex_BA24.csv
│   ├── Brain_Frontal_Cortex_BA9.csv
│   ├── Brain_Hippocampus.csv
│   ├── Brain_Hypothalamus.csv
│   └── Brain_Nucleus_accumbens_basal_ganglia.csv
├── bip/
│   ├── Brain_Amygdala.csv
│   └── ...
├── ocd/
│   ├── Brain_Amygdala.csv
│   └── ...
└── ptsd/
    ├── Brain_Amygdala.csv
    └── ...
```

### Example file content

```csv
gene,gene_name,zscore,effect_size,pvalue
ENSG00000149295,DRD2,-2.31,0.045,0.021
ENSG00000102468,HTR2A,1.87,0.032,0.061
ENSG00000176884,GRIN1,0.54,0.011,0.589
```

Only the `gene_name` and `zscore` columns are used by the pipeline. Additional columns are ignored.

---

## Usage

### Command Line

```bash
python psychiatric_proximity.py \
    --mdd data/mdd/ \
    --bip data/bip/ \
    --ocd data/ocd/ \
    --new data/ptsd/ \
    --label PTSD \
    --out results/
```

#### All options

| Flag | Description | Default |
|------|-------------|---------|
| `--mdd` | Folder containing MDD S-PrediXcan result files | *(required)* |
| `--bip` | Folder containing BIP S-PrediXcan result files | *(required)* |
| `--ocd` | Folder containing OCD S-PrediXcan result files | *(required)* |
| `--new` | Folder containing new-disease S-PrediXcan result files | *(required)* |
| `--label` | Display label for the new disease | `NEW` |
| `--out` | Output directory (created if it does not exist) | `results` |
| `--min-genes` | Minimum number of genes required per pathway to compute a score | `2` |
| `-v, --verbose` | Enable debug-level logging | off |
| `--version` | Print version and exit | — |

### Python API

The pipeline can also be imported and called programmatically:

```python
from psychiatric_proximity import run_pipeline

results = run_pipeline(
    mdd_folder="data/mdd/",
    bip_folder="data/bip/",
    ocd_folder="data/ocd/",
    new_folder="data/ptsd/",
    label="PTSD",
    out_dir="results/",
    min_genes=2,
)

# Returned dictionary contains all computed data structures
meta_z      = results["meta_z"]        # dict of {disease: {gene: z}}
pw_detail   = results["pw_detail"]     # detailed pathway scores
pw_broad    = results["pw_broad"]      # broad pathway scores
prox_detail = results["prox_detail"]   # proximity table (23 pathways)
prox_broad  = results["prox_broad"]    # proximity table (13 pathways)
gene_prox   = results["gene_prox"]     # gene-level proximity
```

---

## Output

All output files are written to the directory specified by `--out`.

### Tables

| File | Description |
|------|-------------|
| `pathway_scores.csv` | Stouffer Z, mean \|Z\|, gene count, and coverage for every pathway × disease combination at both detailed and broad resolution |
| `proximity_detailed.csv` | Pearson r, Spearman ρ, cosine similarity, and Euclidean distance between the new disease and each reference axis using 23 detailed pathways |
| `proximity_broad.csv` | Same proximity metrics using 13 broad pathway categories |
| `proximity_gene_level.csv` | Pearson r, Spearman ρ, and cosine similarity computed directly on shared curated-gene meta-Z values |
| `meta_z_all_genes.csv` | Per-gene Stouffer meta-Z values for every disease, useful for downstream gene-level analyses |
| `summary.txt` | Human-readable summary identifying the most proximal reference axis at each resolution |

### Figures

| File | Description |
|------|-------------|
| `heatmap_detailed.png` | Pathway × disease heatmap (23 pathways) with diverging colour scale centred at zero |
| `heatmap_broad.png` | Pathway × disease heatmap (13 pathways) |
| `radar_detailed.png` | Overlaid spider chart showing pathway Stouffer Z profiles for all four diseases (23 pathways) |
| `radar_broad.png` | Spider chart (13 pathways) |
| `pca_detailed.png` | PCA biplot projecting all four diseases into the first two principal components of pathway space (23 pathways) |
| `pca_broad.png` | PCA biplot (13 pathways) |
| `ternary_detailed.png` | Ternary diagram positioning the new disease within the MDD–BIP–OCD triangle based on cosine similarity (23 pathways) |
| `ternary_broad.png` | Ternary diagram (13 pathways) |
| `proximity_bars.png` | Grouped bar chart comparing Pearson r, Spearman ρ, and cosine similarity across all three reference axes at detailed, broad, and gene-level resolution |

---

## Methods

### Meta-Analysis Across Brain Regions

For each disease, the pipeline reads S-PrediXcan z-scores from all available brain-region files. For every curated gene appearing in one or more regions, it computes a Stouffer meta-Z:

$$
Z_{\text{meta}} = \frac{\sum_{i=1}^{k} z_i}{\sqrt{k}}
$$

where \(k\) is the number of brain regions in which the gene has a valid z-score. This yields a single summary statistic per gene per disease.

### Pathway Scoring

For each pathway (a predefined set of \(N\) genes), the pipeline identifies the \(n \leq N\) genes with available meta-Z values and computes two scores:

**Directional enrichment (Stouffer Z):**

$$
Z_{\text{pathway}} = \frac{\sum_{i=1}^{n} Z_{\text{meta},i}}{\sqrt{n}}
$$

This captures whether the pathway is systematically up- or down-regulated.

**Dysregulation magnitude:**

$$
\overline{|Z|}_{\text{pathway}} = \frac{1}{n} \sum_{i=1}^{n} |Z_{\text{meta},i}|
$$

This captures the overall strength of association regardless of direction.

Pathways with fewer than `--min-genes` genes with available data are assigned NaN scores.

### Proximity Metrics

The pipeline represents each disease as a vector of pathway scores (either 23- or 13-dimensional) and computes four metrics between the new disease vector \(\mathbf{a}\) and each reference axis vector \(\mathbf{b}\):

**Pearson correlation** measures linear agreement in pathway scores. **Spearman correlation** measures rank-order agreement, robust to outlier pathways. **Cosine similarity** measures the angle between the two vectors in pathway space, capturing profile shape regardless of magnitude. **Euclidean distance** measures the absolute separation in pathway space, sensitive to differences in both shape and magnitude.

At the gene level, the pipeline computes Pearson, Spearman, and cosine similarity directly on the shared curated-gene meta-Z vectors between the new disease and each reference axis.

### Ternary Projection

The ternary diagram positions the new disease within an equilateral triangle whose vertices represent MDD, BIP, and OCD. Cosine similarity values are shifted to ensure all are non-negative, then normalised to sum to 1 to obtain barycentric coordinates. A disease with equal similarity to all three axes appears at the centroid; one strongly resembling a single axis appears near that vertex.

---

## Example

Running the pipeline with PTSD as the new disease:

```bash
python psychiatric_proximity.py \
    --mdd data/mdd/ \
    --bip data/bip/ \
    --ocd data/ocd/ \
    --new data/ptsd/ \
    --label PTSD \
    --out results_ptsd/
```

The summary output will report which reference axis PTSD most closely resembles:

```
================================================================
 PROXIMITY SUMMARY  –  PTSD
 Pipeline version 1.0.0
================================================================

DETAILED (23 pathways):
  Most proximal axis : MDD
    Pearson r   = 0.7234   (p = 1.05e-04)
    Spearman ρ  = 0.6891   (p = 2.87e-04)
    Cosine sim  = 0.7518
    Euclidean   = 4.2103
    Pathways    = 23

...
================================================================
```

*(Values shown are illustrative and do not represent real results.)*

---

## Limitations

**Independence assumption.** Stouffer's method assumes independent z-scores. Brain-region S-PrediXcan models use region-specific eQTL weights but may share overlapping SNPs, so z-scores across regions are not fully independent. The meta-Z should be interpreted as a convenient summary statistic rather than a formal hypothesis test.

**Fixed reference axes.** The pipeline uses MDD, BIP, and OCD as the three reference axes. Disorders that do not lie along any of these axes will appear near the centroid of the ternary plot, which could reflect either genuine equidistance or poor coverage of the relevant biology.

**Gene-set coverage.** S-PrediXcan only imputes expression for genes with sufficiently heritable cis-regulated expression in a given tissue. Pathways containing many genes without valid prediction models will have reduced coverage, potentially biasing pathway scores.

**Ternary coordinates are relative.** The barycentric projection normalises similarity values, so the ternary position reflects the relative ranking of proximities rather than their absolute magnitudes. Two diseases can occupy the same ternary position with very different absolute similarity values.

**Gene-set currency.** The curated gene sets are frozen at the time of pipeline release. As pathway databases and cell-type marker lists are updated, the gene sets may need revision.

---

## Citation

If you use this pipeline in your research, please cite:

> Cheung, N. twas-axis. 2026. Available at: https://github.com/cheungngo/twas-axis

Additionally, please cite the tools and resources on which this pipeline depends:

- **S-PrediXcan:** Barbeira AN et al. "Exploring the phenotypic consequences of tissue specific gene expression variation inferred from GWAS summary statistics." *Nature Communications* 9, 1825 (2018).
- **MSigDB gene sets:** Liberzon A et al. "The Molecular Signatures Database hallmark gene set collection." *Cell Systems* 1(6), 417–425 (2015).
- **Cell-type markers:** Zhang Y et al. "Purification and characterization of progenitor and mature human astrocytes reveals transcriptional and functional differences with mouse." *Neuron* 89(1), 37–53 (2016).

