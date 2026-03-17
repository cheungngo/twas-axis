#!/usr/bin/env python3
"""
=============================================================================
 PSYCHIATRIC DISEASE PROXIMITY PIPELINE  (S-PrediXcan / TWAS)
=============================================================================
 Three-axis projection: MDD  /  BIP  /  OCD
 Projects a new disease into this space using 13 curated neurobiological
 gene-set categories (23 pathways when monoamine subcategories are expanded).

 Input  : One folder per disease, each containing S-PrediXcan result
          files (one per brain region) as CSV/TSV with columns:
              gene_name   zscore   [pvalue]
 Output : Proximity CSVs, pathway score tables, PCA / radar / heatmap /
          ternary / bar plots.

 Usage
 -----
   python psychiatric_proximity.py \\
       --mdd data/mdd/ --bip data/bip/ --ocd data/ocd/ \\
       --new data/ptsd/ --label PTSD --out results/

 Citation
 --------
   <Your name / lab>, "Psychiatric proximity pipeline", 2025.
   Gene sets sourced from MSigDB (GOBP, KEGG, Reactome),
   Zhang et al. 2016 (cell-type markers), and HGNC (HLA complex).
=============================================================================
"""

__version__ = "1.0.0"

import argparse
import logging
import os
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless backend – must precede pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Suppress only known noisy warnings (e.g. matplotlib font cache)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ===================================================================
#  1.  CURATED GENE SETS
# ===================================================================
# Sources:
#   Monoamine sub-categories        – literature curation (receptors,
#                                      enzymes, transporters per NT system)
#   GOBP_Reg_Synaptic_Plasticity    – GO:0048167 via MSigDB v2024.1
#   KEGG_LTP                        – hsa04720 via MSigDB
#   GOBP_Glutamatergic              – GO:0035249 via MSigDB
#   GOBP_Synapse_Pruning            – GO:0098883 via MSigDB
#   Reactome_Complement             – R-HSA-166658 via MSigDB
#   Reactome_Senescence             – R-HSA-2559583 via MSigDB
#   Reactome_Telomere               – R-HSA-157579 via MSigDB
#   Adult_Astrocyte / Oligodendro-
#     cyte / Microglia markers      – Zhang et al., Neuron 2016
#   HLA_Complex                     – HGNC gene group 588
# ===================================================================

MONOAMINES = {
    "Dopamine_Receptors": [
        "DRD1","DRD2","DRD3","DRD4","DRD5"],
    "Dopamine_Synthesis_Metabolism": [
        "TH","DDC","COMT","MAOA","MAOB","DBH","GCH1","PTS","QDPR","SPR"],
    "Dopamine_Transport_Signaling": [
        "SLC6A3","SLC18A1","SLC18A2","PPP1R1B","GNAL","ADCY5","PDE1B","PDE10A"],
    "Serotonin_Receptors": [
        "HTR1A","HTR1B","HTR1D","HTR1E","HTR1F","HTR2A","HTR2B","HTR2C",
        "HTR3A","HTR3B","HTR3C","HTR3D","HTR3E","HTR4","HTR5A","HTR5BP",
        "HTR6","HTR7"],
    "Serotonin_Synthesis_Metabolism": [
        "TPH1","TPH2","SLC6A4","AANAT","ASMT"],
    "Norepinephrine_Receptors": [
        "ADRA1A","ADRA1B","ADRA1D","ADRA2A","ADRA2B","ADRA2C",
        "ADRB1","ADRB2","ADRB3"],
    "Norepinephrine_Synthesis_Transport": [
        "PNMT","SLC6A2"],
    "Histamine_System": [
        "HRH1","HRH2","HRH3","HRH4","HDC","HNMT","SLC22A3"],
    "Trace_Amine_Receptors": [
        "TAAR1","TAAR2","TAAR5","TAAR6","TAAR8","TAAR9"],
    "Downstream_Signaling": [
        "GNB1","GNB2","GNB3","GNB4","GNB5","GNG2","GNG3","GNG4","GNG7",
        "GNG11","GNG12","ADCY1","ADCY2","ADCY7","ADCY8","ADCY9",
        "PDE4A","PDE4B","PDE4C","PDE4D"],
    "Vesicular_Machinery": [
        "SYT1","SYT2","SYT4","SYT7","SYT11","SNAP25","VAMP2",
        "STX1A","STX1B","CPLX1","CPLX2"],
}

GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY_ALL = [
    "ABHD6","ABL1","ACE","ACP4","ADCY1","ADCY8","ADGRB1","ADORA1","ADORA2A",
    "AGER","AKAP5","ANAPC2","APOE","APP","ARC","ARF1","ATF4","BAIAP2","BCL7A",
    "BEST1","BRAF","BRSK1","C22orf39","CALB1","CALB2","CALHM2","CALM1","CALM2",
    "CALM3","CALN1","CAMK2A","CAMK2B","CAMK2D","CAMK2G","CBLN1","CD2AP","CD38",
    "CDC20","CDK5","CFL1","CHRDL1","CHRNA2","CHRNA7","CLN3","CNTN2","CNTN4",
    "CPEB3","CPLX2","CREB1","CRH","CRHR2","CX3CL1","CX3CR1","CYP46A1","DAG1",
    "DBN1","DLG4","DRD1","DRD2","DRD5","EIF2AK4","EIF4EBP2","ELOVL4","EPHA4",
    "EPHB2","F2R","FAM107A","FMR1","FXR1","FXR2","GFAP","GIPC1","GRIA1","GRIA3",
    "GRID1","GRID2","GRID2IP","GRIK2","GRIN1","GRIN2A","GRIN2B","GRIN2C","GRIN2D",
    "GRIN3A","GRIN3B","GRM2","GRM5","GSG1L","GSK3B","HMGCR","HRAS","HRH1",
    "IGSF11","INS","IQSEC2","ITPKA","ITPR3","JPH3","JPH4","KAT2A","KCNB1",
    "KCNJ10","KCNQ3","KIT","KMT2A","KRAS","LARGE1","LGMN","LILRB2","LRRTM1",
    "LRRTM2","LYPD6","LZTS1","MAP1A","MAP1B","MAPK1","MAPT","MCTP1","MECP2",
    "MEF2C","MIR30B","MIR320A","MIR320B1","MIR320B2","MIR320C1","MIR320C2",
    "MIR320D1","MIR320D2","MIR320E","MIR324","MIR337","MIR342","MIR421",
    "MIR433","MIR541","MIR545","MIR95","MME","MPP2","NCDN","NCSTN","NETO1",
    "NEURL1","NEUROD2","NF1","NFATC4","NOG","NPAS4","NPTN","NR2E1","NRGN",
    "NSG1","NSMF","NTRK2","P2RX3","PAIP2","PDE9A","PENK","PICK1","PLK2",
    "PPFIA3","PPP3CB","PRKAR1B","PRKCG","PRKCZ","PRNP","PRRT1","PRRT2",
    "PSEN1","PTK2B","PTN","RAB11A","RAB3A","RAB3GAP1","RAB5A","RAB8A","RAC1",
    "RAPGEF2","RARA","RASGRF1","RASGRF2","RELN","RGS14","S100B","SCT","SCTR",
    "SERPINE2","SHANK2","SHANK3","SHISA6","SHISA7","SHISA8","SHISA9","SIPA1L1",
    "SLC18A3","SLC1A1","SLC24A1","SLC24A2","SLC38A1","SLC4A10","SLC8A2",
    "SLC8A3","SLITRK4","SNAP25","SNCA","SORCS2","SORCS3","SQSTM1","SRF","SSH1",
    "STAU1","STAU2","STX3","STX4","STXBP1","SYAP1","SYNGAP1","SYNGR1","SYP",
    "SYT12","SYT4","SYT7","TNR","TSHZ3","TYROBP","UBE3A","UNC13C","VAMP2",
    "VGF","VPS13A","YTHDF1","YWHAG","YWHAH","ZDHHC2"]

KEGG_LTP_ALL = [
    "ADCY1","ADCY8","ARAF","ATF4","BRAF","CACNA1C","CALM1","CALM2","CALM3",
    "CALML3","CALML5","CALML6","CAMK2A","CAMK2B","CAMK2D","CAMK2G","CAMK4",
    "CHP1","CHP2","CREBBP","EP300","GNAQ","GRIA1","GRIA2","GRIN1","GRIN2A",
    "GRIN2B","GRIN2C","GRIN2D","GRM1","GRM5","HRAS","ITPR1","ITPR2","ITPR3",
    "KRAS","MAP2K1","MAP2K2","MAPK1","MAPK3","NRAS","PLCB1","PLCB2","PLCB3",
    "PLCB4","PPP1CA","PPP1CB","PPP1CC","PPP1R12A","PPP1R1A","PPP3CA","PPP3CB",
    "PPP3CC","PPP3R1","PPP3R2","PRKACA","PRKACB","PRKACG","PRKCA","PRKCB",
    "PRKCG","PRKX","RAF1","RAP1A","RAP1B","RAPGEF3","RPS6KA1","RPS6KA2",
    "RPS6KA3","RPS6KA6"]

GOBP_GLUTAMATERGIC_ALL = [
    "ABCC8","ABTB3","ADORA1","ADORA2A","ALS2","ATAD1","ATP1A2","CACNB4",
    "CACNG2","CACNG3","CACNG4","CACNG5","CACNG7","CACNG8","CCL2","CCR2","CDH8",
    "CDK5","CLCN3","CLN3","CLSTN3","CNIH2","CNIH3","DISC1","DKK1","DRD1","DRD2",
    "DRD3","DSCAM","DTNBP1","EXT1","FRRS1L","FXR1","GRIA1","GRIA2","GRIA3",
    "GRIA4","GRID1","GRID2","GRIK1","GRIK2","GRIK3","GRIK4","GRIK5","GRIN1",
    "GRIN2A","GRIN2B","GRIN2C","GRIN2D","GRIN3A","GRIN3B","GRM1","GRM2","GRM3",
    "GRM4","GRM5","GRM6","GRM7","GRM8","HCN1","HOMER1","HOMER2","HOMER3",
    "HTR2A","IQSEC2","KCNJ8","KMO","LRRK2","MAPK8IP2","MEF2C","MIR142","NAPA",
    "NAPB","NF1","NLGN1","NLGN2","NLGN3","NPS","NPY2R","NR3C1","NRXN1","NTRK1",
    "OPHN1","P2RX1","PLPPR4","PNOC","PRKN","PSEN1","PTK2B","RAB3GAP1","RELN",
    "RNF167","SERPINE2","SHANK3","SLC17A6","SLC17A7","SLC17A8","SLC1A4",
    "SLC38A2","STXBP1","SYT1","TNF","TNR","TPRG1L","TSHZ3","UCN","UNC13A",
    "UNC13B","UNC13C","VPS54"]

GOBP_SYNAPSE_PRUNING_ALL = [
    "ADGRB3","C1QA","C1QB","C1QC","C1QL1","C3","CX3CL1","CX3CR1","DKK1",
    "EPHA4","ITGAM","ITGB1","KCNK13","NGEF","PLXNC1","SARM1","TREM2","VANGL2"]

REACTOME_COMPLEMENT_ALL = [
    "C1QA","C1QB","C1QC","C1R","C1S","C2","C3","C3AR1","C4A","C4B","C4B_2",
    "C4BPA","C4BPB","C5","C5AR1","C5AR2","C6","C7","C8A","C8B","C8G","C9",
    "CD19","CD46","CD55","CD59","CD81","CFB","CFD","CFH","CFHR1","CFHR2",
    "CFHR3","CFHR4","CFHR5","CFI","CFP","CLU","COLEC10","COLEC11","CPB2",
    "CPN1","CPN2","CR1","CR2","CRP","ELANE","F2","FCN1","FCN2","FCN3","GZMM",
    "IGHG1","IGHG2","IGHG4","IGHV1-2","IGHV1-46","IGHV1-69","IGHV2-5",
    "IGHV2-70","IGHV3-11","IGHV3-13","IGHV3-23","IGHV3-30","IGHV3-33",
    "IGHV3-48","IGHV3-53","IGHV3-7","IGHV4-34","IGHV4-39","IGHV4-59",
    "IGKV1-12","IGKV1-16","IGKV1-17","IGKV1-33","IGKV1-39","IGKV1-5",
    "IGKV1D-12","IGKV1D-16","IGKV1D-33","IGKV1D-39","IGKV2-28","IGKV2-30",
    "IGKV2D-28","IGKV2D-30","IGKV2D-40","IGKV3-11","IGKV3-15","IGKV3-20",
    "IGKV3D-20","IGKV4-1","IGKV5-2","IGLC2","IGLC3","IGLV1-40","IGLV1-44",
    "IGLV1-47","IGLV1-51","IGLV2-11","IGLV2-14","IGLV2-23","IGLV2-8",
    "IGLV3-1","IGLV3-19","IGLV3-21","IGLV3-25","IGLV3-27","IGLV6-57",
    "IGLV7-43","MASP1","MASP2","MBL2","PROS1","SERPING1","VTN"]

REACTOME_SENESCENCE_ALL = [
    "ACD","AGO1","AGO3","AGO4","ANAPC1","ANAPC10","ANAPC11","ANAPC15","ANAPC16",
    "ANAPC2","ANAPC4","ANAPC5","ANAPC7","ASF1A","ATM","BMI1","CABIN1","CBX2",
    "CBX4","CBX6","CBX8","CCNA1","CCNA2","CCNE1","CCNE2","CDC16","CDC23","CDC26",
    "CDC27","CDK2","CDK4","CDK6","CDKN1A","CDKN1B","CDKN2A","CDKN2B","CDKN2C",
    "CDKN2D","CEBPB","CXCL8","E2F1","E2F2","E2F3","EED","EHMT1","EHMT2","EP400",
    "ERF","ETS1","ETS2","EZH2","FOS","FZR1","H1-0","H1-1","H1-2","H1-3","H1-4",
    "H1-5","H2AB1","H2AC14","H2AC18","H2AC19","H2AC20","H2AC4","H2AC6","H2AC7",
    "H2AC8","H2AJ","H2AX","H2AZ2","H2BC1","H2BC10","H2BC11","H2BC12","H2BC12L",
    "H2BC13","H2BC14","H2BC15","H2BC17","H2BC21","H2BC26","H2BC3","H2BC4",
    "H2BC5","H2BC6","H2BC7","H2BC8","H2BC9","H3-3A","H3-3B","H3-4","H3C1",
    "H3C10","H3C11","H3C12","H3C13","H3C14","H3C15","H3C2","H3C3","H3C4",
    "H3C6","H3C7","H3C8","H4C1","H4C11","H4C12","H4C13","H4C14","H4C15",
    "H4C16","H4C2","H4C3","H4C4","H4C5","H4C6","H4C8","H4C9","HIRA","HMGA1",
    "HMGA2","ID1","IFNB1","IGFBP7","IL1A","IL6","JUN","KAT5","KDM6B","LMNB1",
    "MAP2K3","MAP2K4","MAP2K6","MAP2K7","MAP3K5","MAP4K4","MAPK1","MAPK10",
    "MAPK11","MAPK14","MAPK3","MAPK7","MAPK8","MAPK9","MAPKAPK2","MAPKAPK3",
    "MAPKAPK5","MDM2","MDM4","MINK1","MIR24-1","MIR24-2","MOV10","MRE11","NBN",
    "NFKB1","PHC1","PHC2","PHC3","POT1","RAD50","RB1","RBBP4","RBBP7","RELA",
    "RING1","RNF2","RPS27A","RPS6KA1","RPS6KA2","RPS6KA3","SCMH1","SP1","STAT3",
    "SUZ12","TERF1","TERF2","TERF2IP","TFDP1","TFDP2","TINF2","TNIK","TNRC6A",
    "TNRC6B","TNRC6C","TP53","TXN","UBA52","UBB","UBC","UBE2C","UBE2D1",
    "UBE2E1","UBE2S","UBN1","VENTX"]

REACTOME_TELOMERE_ALL = [
    "ACD","ANKRD28","ATRX","BLM","CCNA1","CCNA2","CDK2","CHTF18","CHTF8","CTC1",
    "DAXX","DKC1","DNA2","DSCC1","FEN1","GAR1","H2AB1","H2AC14","H2AC18",
    "H2AC19","H2AC20","H2AC4","H2AC6","H2AC7","H2AC8","H2AJ","H2AX","H2AZ2",
    "H2BC1","H2BC10","H2BC11","H2BC12","H2BC12L","H2BC13","H2BC14","H2BC15",
    "H2BC17","H2BC21","H2BC26","H2BC3","H2BC4","H2BC5","H2BC6","H2BC7","H2BC8",
    "H2BC9","H3-3A","H3-3B","H3-4","H4C1","H4C11","H4C12","H4C13","H4C14",
    "H4C15","H4C16","H4C2","H4C3","H4C4","H4C5","H4C6","H4C8","H4C9","LIG1",
    "NHP2","NOP10","PCNA","PIF1","POLA1","POLA2","POLD1","POLD2","POLD3",
    "POLD4","POLR2A","POLR2B","POLR2C","POLR2D","POLR2E","POLR2F","POLR2G",
    "POLR2H","POLR2I","POLR2J","POLR2K","POLR2L","POT1","PPP6C","PPP6R3",
    "PRIM1","PRIM2","RFC1","RFC2","RFC3","RFC4","RFC5","RPA1","RPA2","RPA3",
    "RTEL1","RUVBL1","RUVBL2","SHQ1","STN1","TEN1","TERF1","TERF2","TERF2IP",
    "TERT","TINF2","WRAP53","WRN"]

ADULT_ASTRO_ALL = [
    "ALDH1L1","EYA1","NWD1","SLCO1C1","HGF","SPARCL1","SLC25A18","FGFR3",
    "SDC4","GFAP","ACSBG1","PRODH","ITGB4","SLC7A11","AGT","PPP1R3C","RANBP3L",
    "RNF43","GJB6","SLC14A1","GJA1","USP9Y","TTTY15","KDM5D","SLC39A12",
    "COL16A1","DLC1","ABO","ABCC9","MGST1","GABRG1","ITGA7","CLTCL1","DCHS2",
    "RGS9","SLC30A10","GPR98","BMPR1B","SLC4A4","GRM3"]

ADULT_OLIGO_ALL = [
    "MSX1","ADAMTS4","TF","TMEM144","ENPP2","KLK6","ERMN","RNASE1","PLP1",
    "OPALIN","CNDP1","FOLH1","UGT8","CNP","PLEKHB1","MAL","GPR37","NR4A2",
    "B3GNT7","GPNMB","GLYCTK","GJB1","PAQR6","GSN","ABCA8","FA2H","CARNS1",
    "MOBP","MBP","CLDN11","RRBP1","OLFML2B","KCNMB4","LGI3","MAG"]

ADULT_MICRO_ALL = [
    "ST8SIA4","KIAA0226L","CSF1R","LAPTM5","CD14","RGS10","PTPRC","C3","C3AR1",
    "ALOX5AP","HLA-DRA","BCL2A1","PLEK","ITGAX","DHRS9","OLR1","CD53","LCP1",
    "CD74","CD83","A2M","GAB3","CMKLR1","TREM1","GALR1"]

HLA_COMPLEX_ALL = [
    "HLA-A","HLA-B","HLA-C","HLA-DMA","HLA-DMB","HLA-DOA","HLA-DOB","HLA-DPA1",
    "HLA-DPA2","HLA-DPA3","HLA-DPB1","HLA-DPB2","HLA-DQA1","HLA-DQA2",
    "HLA-DQB1","HLA-DQB2","HLA-DQB3","HLA-DRA","HLA-DRB1","HLA-DRB2",
    "HLA-DRB3","HLA-DRB4","HLA-DRB5","HLA-DRB6","HLA-DRB7","HLA-DRB8",
    "HLA-DRB9","HLA-E","HLA-F","HLA-G","HLA-H","HLA-J","HLA-K","HLA-L",
    "HLA-N","HLA-P","HLA-S","HLA-T","HLA-U","HLA-V","HLA-W","HLA-X","HLA-Y",
    "HLA-Z"]


# ===================================================================
#  2.  BUILD PATHWAY DICTIONARIES
# ===================================================================

# Detailed: 23 pathways (11 monoamine sub-categories + 12 others)
PATHWAY_DETAIL = {}
for _subcat_name, _subcat_genes in MONOAMINES.items():
    PATHWAY_DETAIL[f"Mono_{_subcat_name}"] = _subcat_genes

PATHWAY_DETAIL["GOBP_Reg_Synaptic_Plasticity"]  = GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY_ALL
PATHWAY_DETAIL["KEGG_LTP"]                       = KEGG_LTP_ALL
PATHWAY_DETAIL["GOBP_Glutamatergic"]             = GOBP_GLUTAMATERGIC_ALL
PATHWAY_DETAIL["GOBP_Synapse_Pruning"]           = GOBP_SYNAPSE_PRUNING_ALL
PATHWAY_DETAIL["Reactome_Complement"]            = REACTOME_COMPLEMENT_ALL
PATHWAY_DETAIL["Reactome_Senescence"]            = REACTOME_SENESCENCE_ALL
PATHWAY_DETAIL["Reactome_Telomere"]              = REACTOME_TELOMERE_ALL
PATHWAY_DETAIL["Adult_Astrocyte"]                = ADULT_ASTRO_ALL
PATHWAY_DETAIL["Adult_Oligodendrocyte"]          = ADULT_OLIGO_ALL
PATHWAY_DETAIL["Adult_Microglia"]                = ADULT_MICRO_ALL
PATHWAY_DETAIL["HLA_Complex"]                    = HLA_COMPLEX_ALL

# Broad: 13 pathways (aggregate monoamines + 12 others)
PATHWAY_BROAD = {
    "Monoamines_ALL": sorted({g for gs in MONOAMINES.values() for g in gs})
}
for _k, _v in PATHWAY_DETAIL.items():
    if not _k.startswith("Mono_"):
        PATHWAY_BROAD[_k] = _v

# Master gene universe
ALL_GENES_SET = frozenset(g for gs in PATHWAY_DETAIL.values() for g in gs)


# ===================================================================
#  3.  CORE FUNCTIONS
# ===================================================================

def _validate_folder(path, tag):
    """Ensure folder exists and contains at least one data file."""
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(
            f"Folder for {tag} does not exist: {path}\n"
            f"  Expected a directory containing S-PrediXcan CSV/TSV files."
        )
    data_files = []
    for ext in ("*.csv", "*.txt", "*.tsv", "*.dat"):
        data_files.extend(p.glob(ext))
    if not data_files:
        raise FileNotFoundError(
            f"Folder for {tag} contains no data files: {path}\n"
            f"  Expected .csv / .tsv / .txt / .dat files with columns: "
            f"gene_name, zscore"
        )
    return p


def load_spredixcan_folder(folder_path):
    """
    Read every S-PrediXcan result file in *folder_path* (one per brain
    region).  For each curated gene, collect z-scores across regions and
    meta-analyse with Stouffer's method::

        meta_Z = sum(z_i) / sqrt(k)

    Returns
    -------
    dict : {gene_name: meta_z}

    Notes
    -----
    Stouffer's method assumes independent z-scores.  Brain-region
    S-PrediXcan models use region-specific eQTL weights, so the
    resulting z-scores are not fully independent.  The meta-Z should
    be interpreted as a summary statistic, not a formal p-value.
    """
    folder = Path(folder_path)
    files = []
    for ext in ("*.csv", "*.txt", "*.tsv", "*.dat"):
        files.extend(folder.glob(ext))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No result files found in {folder_path}")

    log.info("  %d file(s) in %s/", len(files), folder.name)

    gene_z = defaultdict(list)

    for f in files:
        df = None
        suffix = f.suffix.lower()
        sep_first  = "," if suffix == ".csv" else "\t"
        sep_second = "\t" if suffix == ".csv" else ","

        for sep in [sep_first, sep_second, r"\s+"]:
            try:
                candidate = pd.read_csv(
                    f, sep=sep,
                    engine="python" if sep == r"\s+" else "c",
                )
                if ("gene_name" in candidate.columns
                        and "zscore" in candidate.columns):
                    df = candidate
                    break
            except Exception:
                continue

        if df is None:
            log.warning("    SKIP %s – could not find gene_name/zscore columns",
                        f.name)
            continue

        sub = df[["gene_name", "zscore"]].copy()
        if "pvalue" in df.columns:
            sub["pvalue"] = df["pvalue"]
            sub = sub.dropna(subset=["gene_name", "zscore"])
            sub = sub.sort_values("pvalue").drop_duplicates(
                "gene_name", keep="first"
            )
        else:
            sub = sub.dropna().drop_duplicates("gene_name", keep="first")

        n_hits = 0
        for _, row in sub.iterrows():
            gn = str(row["gene_name"]).strip()
            if gn in ALL_GENES_SET:
                gene_z[gn].append(float(row["zscore"]))
                n_hits += 1
        log.info("    %s: %d genes read, %d curated", f.name, len(sub), n_hits)

    if not gene_z:
        raise RuntimeError(
            f"No curated genes found in any file in {folder_path}.\n"
            f"  Check that your files contain 'gene_name' and 'zscore' columns\n"
            f"  and that gene names match HGNC symbols (e.g. DRD2, HTR2A, GRIN1)."
        )

    # Stouffer meta-Z across brain regions
    meta = {g: np.sum(zs) / np.sqrt(len(zs)) for g, zs in gene_z.items()}
    regions_per_gene = np.mean([len(v) for v in gene_z.values()])
    log.info("  → meta-Z for %d curated genes  (%.1f regions/gene avg)",
             len(meta), regions_per_gene)
    return meta


def pathway_scores(gene_meta_z, pathway_dict, min_genes=2):
    """
    For each pathway compute:

    - stouffer_z : sum(z_i) / sqrt(n)   – directional enrichment
    - mean_abs_z : mean(|z_i|)          – dysregulation magnitude
    - coverage   : n / N                – fraction of pathway genes found
    """
    out = {}
    for pw, genes in pathway_dict.items():
        zv = [gene_meta_z[g] for g in genes if g in gene_meta_z]
        n = len(zv)
        if n >= min_genes:
            out[pw] = dict(
                stouffer_z = np.sum(zv) / np.sqrt(n),
                mean_abs_z = np.mean(np.abs(zv)),
                n_genes    = n,
                n_total    = len(genes),
                coverage   = round(100 * n / len(genes), 1),
            )
        else:
            out[pw] = dict(
                stouffer_z=np.nan, mean_abs_z=np.nan,
                n_genes=n, n_total=len(genes),
                coverage=round(100 * n / len(genes), 1),
            )
    return out


def _vec(scores, pathways, metric="stouffer_z"):
    """Extract a numeric vector from pathway score dicts."""
    return np.array([scores[p][metric] for p in pathways])


def proximity_table(ref_dict, new_scores, pw_dict, metric="stouffer_z"):
    """
    Correlation / cosine / Euclidean between the new disease vector
    and each reference-axis vector across pathways.
    """
    pws = sorted(pw_dict.keys())
    nv  = _vec(new_scores, pws, metric)
    rows = []
    for name, rscores in ref_dict.items():
        rv    = _vec(rscores, pws, metric)
        valid = ~np.isnan(nv) & ~np.isnan(rv)
        if valid.sum() < 3:
            rows.append(dict(
                Axis=name, Pearson_r=np.nan, Pearson_p="NA",
                Spearman_rho=np.nan, Spearman_p="NA",
                Cosine=np.nan, Euclidean=np.nan, n_pw=int(valid.sum()),
            ))
            continue
        a, b = nv[valid], rv[valid]
        pr, pp = stats.pearsonr(a, b)
        sr, sp = stats.spearmanr(a, b)
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15)
        euc = np.linalg.norm(a - b)
        rows.append(dict(
            Axis=name,
            Pearson_r=round(pr, 4), Pearson_p=f"{pp:.2e}",
            Spearman_rho=round(sr, 4), Spearman_p=f"{sp:.2e}",
            Cosine=round(cos, 4), Euclidean=round(euc, 4),
            n_pw=int(valid.sum()),
        ))
    return pd.DataFrame(rows)


def gene_proximity(ref_z, new_z):
    """Direct gene-level correlation between two meta-Z dictionaries."""
    shared = sorted(set(ref_z) & set(new_z))
    if len(shared) < 5:
        return dict(
            Pearson_r=np.nan, Spearman_rho=np.nan,
            Cosine=np.nan, n_genes=len(shared),
        )
    rv = np.array([ref_z[g] for g in shared])
    nv = np.array([new_z[g] for g in shared])
    pr, _ = stats.pearsonr(rv, nv)
    sr, _ = stats.spearmanr(rv, nv)
    cos = np.dot(rv, nv) / (np.linalg.norm(rv) * np.linalg.norm(nv) + 1e-15)
    return dict(
        Pearson_r=round(pr, 4), Spearman_rho=round(sr, 4),
        Cosine=round(cos, 4), n_genes=len(shared),
    )


# ===================================================================
#  4.  VISUALISATION
# ===================================================================

DISEASE_COLOURS = {"MDD": "#1f77b4", "BIP": "#ff7f0e", "OCD": "#2ca02c"}
NEW_COLOUR      = "#d62728"
DISEASE_MARKERS = {"MDD": "o", "BIP": "s", "OCD": "^"}
NEW_MARKER      = "D"


def _short_label(name, maxlen=22):
    return (name[:maxlen - 1] + "…") if len(name) > maxlen else name


def plot_heatmap(all_scores, pw_dict, labels, metric, outpath):
    """Pathway × disease heatmap with diverging colour scale."""
    pws = sorted(pw_dict.keys())
    mat = np.array([[all_scores[lb][p][metric] for p in pws] for lb in labels])

    fig, ax = plt.subplots(
        figsize=(max(8, len(pws) * 0.45), len(labels) * 0.9 + 2)
    )
    vmax = np.nanmax(np.abs(mat))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pws)))
    ax.set_xticklabels(
        [_short_label(p) for p in pws], rotation=55, ha="right", fontsize=7
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(pws)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(
                    j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 * vmax else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.6, label=f"Pathway {metric}")
    ax.set_title(f"Pathway Scores ({metric})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Heatmap → %s", outpath)


def plot_radar(all_scores, pw_dict, labels, metric, outpath):
    """Overlaid radar / spider chart for all diseases."""
    pws = sorted(pw_dict.keys())
    N = len(pws)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for lb in labels:
        vals = [all_scores[lb][p][metric] for p in pws]
        vals = [v if not np.isnan(v) else 0.0 for v in vals]
        vals += vals[:1]
        col = DISEASE_COLOURS.get(lb, NEW_COLOUR)
        ax.plot(angles, vals, "o-", linewidth=1.8, label=lb,
                color=col, markersize=4)
        ax.fill(angles, vals, alpha=0.08, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([_short_label(p, 18) for p in pws], fontsize=6.5)
    ax.set_title(f"Radar – Pathway {metric}", fontsize=12,
                 fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Radar  → %s", outpath)


def plot_pca(all_scores, pw_dict, labels, metric, outpath):
    """PCA biplot in pathway space (SVD on 4 × P matrix)."""
    pws = sorted(pw_dict.keys())
    mat = np.array([[all_scores[lb][p][metric] for p in pws] for lb in labels])
    mat = np.nan_to_num(mat, nan=0.0)
    centred = mat - mat.mean(axis=0)
    U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    pcs  = U * S
    vexp = (S ** 2 / (S ** 2).sum()) * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, lb in enumerate(labels):
        col = DISEASE_COLOURS.get(lb, NEW_COLOUR)
        mk  = DISEASE_MARKERS.get(lb, NEW_MARKER)
        ax.scatter(pcs[i, 0], pcs[i, 1], s=260, c=col, marker=mk,
                   edgecolors="black", linewidth=1.3, zorder=5, label=lb)
        ax.annotate(lb, (pcs[i, 0], pcs[i, 1]), fontsize=11,
                    fontweight="bold", xytext=(8, 8),
                    textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel(f"PC1  ({vexp[0]:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2  ({vexp[1]:.1f}% var)", fontsize=11)
    ax.set_title("PCA – Curated Pathway Space", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    log.info("  PCA    → %s", outpath)


def _bary_to_xy(w):
    """Barycentric (MDD, BIP, OCD) → Cartesian in equilateral triangle."""
    x = w[1] + 0.5 * w[2]
    y = (np.sqrt(3) / 2) * w[2]
    return x, y


def plot_ternary(prox_df, new_label, outpath, metric_col="Cosine"):
    """
    Position of the new disease in an MDD–BIP–OCD ternary triangle.
    Proximity values are shifted positive and normalised to give
    barycentric coordinates.
    """
    vals = {}
    for _, row in prox_df.iterrows():
        vals[row["Axis"]] = row[metric_col]

    raw = np.array([vals.get("MDD", 0), vals.get("BIP", 0), vals.get("OCD", 0)])
    shifted = raw - raw.min() + 0.01
    weights = shifted / shifted.sum()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")
    ax.axis("off")

    verts = {
        "MDD": np.array([0, 0]),
        "BIP": np.array([1, 0]),
        "OCD": np.array([0.5, np.sqrt(3) / 2]),
    }
    tri = plt.Polygon(list(verts.values()), fill=False,
                       edgecolor="black", lw=2)
    ax.add_patch(tri)

    # gridlines
    for frac in [0.2, 0.4, 0.6, 0.8]:
        for (a, b) in [("MDD", "BIP"), ("BIP", "OCD"), ("OCD", "MDD")]:
            others = [k for k in verts if k not in (a, b)]
            c = others[0]
            p2_start = verts[a] + frac * (verts[c] - verts[a])
            p2_end   = verts[b] + frac * (verts[c] - verts[b])
            ax.plot([p2_start[0], p2_end[0]], [p2_start[1], p2_end[1]],
                    color="grey", lw=0.4, alpha=0.5)

    # vertex labels
    offsets = {"MDD": (-0.06, -0.06), "BIP": (0.02, -0.06),
               "OCD": (-0.02, 0.04)}
    for name, pos in verts.items():
        ox, oy = offsets[name]
        ax.text(pos[0] + ox, pos[1] + oy, name, fontsize=13,
                fontweight="bold", ha="center",
                color=DISEASE_COLOURS[name])
        ax.scatter(*pos, s=180, c=DISEASE_COLOURS[name],
                   marker=DISEASE_MARKERS[name], edgecolors="black",
                   linewidth=1.2, zorder=5)

    nx, ny = _bary_to_xy(weights)
    ax.scatter(nx, ny, s=280, c=NEW_COLOUR, marker=NEW_MARKER,
               edgecolors="black", linewidth=1.5, zorder=6)
    ax.annotate(new_label, (nx, ny), fontsize=11, fontweight="bold",
                xytext=(10, 10), textcoords="offset points",
                color=NEW_COLOUR,
                arrowprops=dict(arrowstyle="->", color=NEW_COLOUR, lw=1.2))

    txt = (f"Barycentric coords ({metric_col}):\n"
           f"  MDD = {weights[0]:.3f}\n"
           f"  BIP = {weights[1]:.3f}\n"
           f"  OCD = {weights[2]:.3f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey",
                      alpha=0.85))

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.05)
    ax.set_title("Ternary Proximity Plot", fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Ternary→ %s", outpath)


def plot_proximity_bars(prox_detail, prox_broad, gene_prox_df,
                        new_label, outpath):
    """Grouped bar chart of similarity metrics for all three resolutions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, df, title in zip(
        axes,
        [prox_detail, prox_broad, gene_prox_df],
        ["Detailed (23 pw)", "Broad (13 pw)", "Gene-level"],
    ):
        axes_names = df["Axis"].tolist()
        x = np.arange(len(axes_names))
        w = 0.25

        pearson  = df["Pearson_r"].fillna(0).values
        spearman = df["Spearman_rho"].fillna(0).values
        cosine   = df["Cosine"].fillna(0).values

        ax.bar(x - w, pearson,  w, label="Pearson r",  color="#4c72b0")
        ax.bar(x,     spearman, w, label="Spearman ρ", color="#55a868")
        ax.bar(x + w, cosine,   w, label="Cosine sim", color="#c44e52")

        ax.set_xticks(x)
        ax.set_xticklabels(axes_names, fontsize=10)
        ax.set_ylabel("Similarity", fontsize=10)
        ax.set_title(f"{new_label} proximity – {title}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Bars   → %s", outpath)


# ===================================================================
#  5.  MAIN PIPELINE
# ===================================================================

def run_pipeline(mdd_folder, bip_folder, ocd_folder, new_folder,
                 label="NEW", out_dir="results", min_genes=2):
    """
    Execute the full psychiatric proximity pipeline.

    Parameters
    ----------
    mdd_folder : str
        Path to folder with MDD S-PrediXcan files (one per brain region).
    bip_folder : str
        Path to folder with BIP S-PrediXcan files.
    ocd_folder : str
        Path to folder with OCD S-PrediXcan files.
    new_folder : str
        Path to folder with new-disease S-PrediXcan files.
    label : str
        Display label for the new disease (default: "NEW").
    out_dir : str
        Output directory (default: "results").
    min_genes : int
        Minimum genes required per pathway to compute a score (default: 2).

    Returns
    -------
    dict
        Keys: 'meta_z', 'pw_detail', 'pw_broad', 'prox_detail',
        'prox_broad', 'gene_prox' — all computed data structures.
    """
    os.makedirs(out_dir, exist_ok=True)
    LBL = label

    n_detail = len(PATHWAY_DETAIL)
    n_broad  = len(PATHWAY_BROAD)
    n_genes  = len(ALL_GENES_SET)
    log.info("=" * 65)
    log.info(" PSYCHIATRIC PROXIMITY PIPELINE  v%s", __version__)
    log.info(" Pathways : %d detailed / %d broad", n_detail, n_broad)
    log.info(" Gene universe : %d unique curated genes", n_genes)
    log.info("=" * 65)

    # ── Validate inputs ──
    folders = [
        ("MDD", mdd_folder), ("BIP", bip_folder),
        ("OCD", ocd_folder), (LBL, new_folder),
    ]
    for tag, fld in folders:
        _validate_folder(fld, tag)

    # ── Load ──
    log.info("[1/5] Loading S-PrediXcan results …")
    meta_z = {}
    for tag, fld in folders:
        log.info("  %s:", tag)
        meta_z[tag] = load_spredixcan_folder(fld)

    # ── Pathway scores ──
    log.info("[2/5] Computing pathway scores …")
    pw_detail = {t: pathway_scores(meta_z[t], PATHWAY_DETAIL, min_genes)
                 for t in meta_z}
    pw_broad  = {t: pathway_scores(meta_z[t], PATHWAY_BROAD, min_genes)
                 for t in meta_z}

    rows = []
    for tag in meta_z:
        for pw, sc in pw_detail[tag].items():
            rows.append(dict(Disease=tag, Pathway=pw,
                             Resolution="detailed", **sc))
        for pw, sc in pw_broad[tag].items():
            rows.append(dict(Disease=tag, Pathway=pw,
                             Resolution="broad", **sc))
    pw_table = pd.DataFrame(rows)
    pw_path = Path(out_dir) / "pathway_scores.csv"
    pw_table.to_csv(pw_path, index=False)
    log.info("  Pathway scores → %s", pw_path)

    # ── Proximity ──
    log.info("[3/5] Computing proximity metrics …")
    ref_d = {k: pw_detail[k] for k in ("MDD", "BIP", "OCD")}
    ref_b = {k: pw_broad[k]  for k in ("MDD", "BIP", "OCD")}

    prox_detail = proximity_table(ref_d, pw_detail[LBL], PATHWAY_DETAIL)
    prox_broad  = proximity_table(ref_b, pw_broad[LBL],  PATHWAY_BROAD)

    gene_rows = []
    for axis in ("MDD", "BIP", "OCD"):
        gp = gene_proximity(meta_z[axis], meta_z[LBL])
        gp["Axis"] = axis
        gene_rows.append(gp)
    gene_prox = pd.DataFrame(gene_rows)[
        ["Axis", "Pearson_r", "Spearman_rho", "Cosine", "n_genes"]
    ]

    log.info("  === DETAILED (23 pathways) ===")
    log.info("\n%s", prox_detail.to_string(index=False))
    log.info("  === BROAD (13 pathways) ===")
    log.info("\n%s", prox_broad.to_string(index=False))
    log.info("  === GENE-LEVEL ===")
    log.info("\n%s", gene_prox.to_string(index=False))

    prox_detail.to_csv(Path(out_dir) / "proximity_detailed.csv", index=False)
    prox_broad.to_csv(Path(out_dir) / "proximity_broad.csv", index=False)
    gene_prox.to_csv(Path(out_dir) / "proximity_gene_level.csv", index=False)

    # ── Plots ──
    log.info("[4/5] Generating plots …")
    all_labels = ["MDD", "BIP", "OCD", LBL]
    od = Path(out_dir)

    plot_heatmap(pw_detail, PATHWAY_DETAIL, all_labels, "stouffer_z",
                 od / "heatmap_detailed.png")
    plot_heatmap(pw_broad,  PATHWAY_BROAD,  all_labels, "stouffer_z",
                 od / "heatmap_broad.png")
    plot_radar(pw_detail, PATHWAY_DETAIL, all_labels, "stouffer_z",
               od / "radar_detailed.png")
    plot_radar(pw_broad,  PATHWAY_BROAD,  all_labels, "stouffer_z",
               od / "radar_broad.png")
    plot_pca(pw_detail, PATHWAY_DETAIL, all_labels, "stouffer_z",
             od / "pca_detailed.png")
    plot_pca(pw_broad,  PATHWAY_BROAD,  all_labels, "stouffer_z",
             od / "pca_broad.png")
    plot_ternary(prox_detail, LBL, od / "ternary_detailed.png", "Cosine")
    plot_ternary(prox_broad,  LBL, od / "ternary_broad.png",   "Cosine")
    plot_proximity_bars(prox_detail, prox_broad, gene_prox, LBL,
                        od / "proximity_bars.png")

    # ── Summary ──
    log.info("[5/5] Writing summary …")

    # Guard against all-NaN cosine columns
    def _safe_idxmax(df, col):
        valid = df[col].dropna()
        if valid.empty:
            return df.iloc[0]
        return df.loc[valid.idxmax()]

    best_detail = _safe_idxmax(prox_detail, "Cosine")
    best_broad  = _safe_idxmax(prox_broad,  "Cosine")
    best_gene   = _safe_idxmax(gene_prox,   "Cosine")

    summary = textwrap.dedent(f"""\
    ================================================================
     PROXIMITY SUMMARY  –  {LBL}
     Pipeline version {__version__}
    ================================================================

    DETAILED (23 pathways):
      Most proximal axis : {best_detail['Axis']}
        Pearson r   = {best_detail['Pearson_r']}   (p = {best_detail.get('Pearson_p', 'NA')})
        Spearman ρ  = {best_detail['Spearman_rho']} (p = {best_detail.get('Spearman_p', 'NA')})
        Cosine sim  = {best_detail['Cosine']}
        Euclidean   = {best_detail.get('Euclidean', 'NA')}
        Pathways    = {best_detail.get('n_pw', 'NA')}

    BROAD (13 pathways):
      Most proximal axis : {best_broad['Axis']}
        Pearson r   = {best_broad['Pearson_r']}   (p = {best_broad.get('Pearson_p', 'NA')})
        Spearman ρ  = {best_broad['Spearman_rho']} (p = {best_broad.get('Spearman_p', 'NA')})
        Cosine sim  = {best_broad['Cosine']}
        Euclidean   = {best_broad.get('Euclidean', 'NA')}
        Pathways    = {best_broad.get('n_pw', 'NA')}

    GENE-LEVEL (all curated genes):
      Most proximal axis : {best_gene['Axis']}
        Pearson r   = {best_gene['Pearson_r']}
        Spearman ρ  = {best_gene['Spearman_rho']}
        Cosine sim  = {best_gene['Cosine']}
        Genes       = {best_gene['n_genes']}

    ================================================================
    """)

    log.info("\n%s", summary)
    summary_path = Path(out_dir) / "summary.txt"
    summary_path.write_text(summary)
    log.info("  Summary → %s", summary_path)

    # Save meta-Z for all genes
    mz_rows = []
    for tag in meta_z:
        for g, z in meta_z[tag].items():
            mz_rows.append(dict(Disease=tag, Gene=g, meta_z=round(z, 6)))
    mz_path = Path(out_dir) / "meta_z_all_genes.csv"
    pd.DataFrame(mz_rows).to_csv(mz_path, index=False)
    log.info("  Gene meta-Z → %s", mz_path)

    log.info("=" * 65)
    log.info(" DONE  –  all outputs in  %s/", out_dir)
    log.info("=" * 65)

    return dict(
        meta_z=meta_z, pw_detail=pw_detail, pw_broad=pw_broad,
        prox_detail=prox_detail, prox_broad=prox_broad,
        gene_prox=gene_prox,
    )


# ===================================================================
#  6.  CLI ENTRY POINT
# ===================================================================

def _build_parser():
    parser = argparse.ArgumentParser(
        prog="psychiatric_proximity",
        description=textwrap.dedent("""\
            Psychiatric Disease Proximity Pipeline (S-PrediXcan / TWAS)

            Projects a new psychiatric phenotype into a three-axis space
            defined by MDD, BIP, and OCD using curated neurobiological
            gene-set signatures derived from brain-region S-PrediXcan.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example
            -------
              python psychiatric_proximity.py \\
                  --mdd data/mdd/ --bip data/bip/ --ocd data/ocd/ \\
                  --new data/ptsd/ --label PTSD --out results/

            Each folder should contain one CSV or TSV per brain region
            with at least the columns: gene_name, zscore
        """),
    )
    parser.add_argument("--mdd", required=True,
                        help="Folder with MDD S-PrediXcan result files")
    parser.add_argument("--bip", required=True,
                        help="Folder with BIP S-PrediXcan result files")
    parser.add_argument("--ocd", required=True,
                        help="Folder with OCD S-PrediXcan result files")
    parser.add_argument("--new", required=True,
                        help="Folder with new-disease S-PrediXcan result files")
    parser.add_argument("--label", default="NEW",
                        help="Display label for the new disease (default: NEW)")
    parser.add_argument("--out", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--min-genes", type=int, default=2,
                        help="Min genes per pathway to compute a score "
                             "(default: 2)")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging")
    return parser


def main():
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        mdd_folder=args.mdd,
        bip_folder=args.bip,
        ocd_folder=args.ocd,
        new_folder=args.new,
        label=args.label,
        out_dir=args.out,
        min_genes=args.min_genes,
    )


if __name__ == "__main__":
    main()
