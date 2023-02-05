# ZINB-based Graph Embedding Autoencoder for Single-cell RNA-seq Interpretations

<p align="center">   
    <a href="https://aaai.org/Conferences/AAAI-22/" alt="Conference">
        <img src="https://img.shields.io/badge/AAAI'22-brightgreen" /></a>
<p/>

Single-cell RNA sequencing (scRNA-seq) provides high-throughput information about the genome-wide gene expression levels at the single-cell resolution, bringing a precise understanding on the transcriptome of individual cells. Unfortunately, the rapidly growing scRNA-seq data and the prevalence of dropout events pose substantial challenges for cell type annotation. Here, we propose a single-cell model-based deep graph embedding clustering (scTAG) method, which simultaneously learns cell–cell topology representations and identifies cell clusters based on deep graph convolutional network. scTAG integrates the zero-inflated negative binomial (ZINB) model into a topology adaptive graph convolutional autoencoder to learn the low-dimensional latent representation and adopts Kullback–Leibler (KL) divergence for the clustering tasks. By simultaneously optimizing the clustering loss, ZINB loss, and the cell graph reconstruction loss, scTAG jointly optimizes cluster label assignment and feature learning with the topological structures preserved in an end-to-end manner. Extensive experiments on 16 single-cell RNA-seq datasets from diverse yet representative single-cell sequencing platforms demonstrate the superiority of scTAG over various state-of-the-art clustering methods.

![fram1 (1)](https://user-images.githubusercontent.com/65069252/144599080-b4762b2e-955a-4411-bd98-a2bff0ad0f82.png)

## Installation

### pip

```
$ pip install -r requirements
```

## Usage

You can run the scTAG from the command line:

```
$ python train.py --dataname Quake_Smart-seq2_Limb_Muscle --highly_genes 500 --pretrain_epochs 1000 --maxiter 300
```

## Arguments

|    Parameter    | Introduction                                                 |
| :-------------: | ------------------------------------------------------------ |
|    dataname     | A h5 file. Contains a matrix of scRNA-seq expression values,true labels, and other information. By default, genes are assumed to be represent-ed by columns and samples are assumed to be represented by rows. |
|  highly genes   | Number of genes selected                                     |
| pretrain epochs | Number of pretrain epochs                                    |
|     maxiter     | Number of training epochs                                    |

## Citation

```
@inproceedings{yu2022zinb,
  title={Zinb-based graph embedding autoencoder for single-cell rna-seq interpretations},
  author={Yu, Zhuohan and Lu, Yifu and Wang, Yunhe and Tang, Fan and Wong, Ka-Chun and Li, Xiangtao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={4},
  pages={4671--4679},
  year={2022}
}
```
