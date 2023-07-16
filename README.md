# ChartT5
[Enhanced Chart Understanding in Vision and Language Task via Cross-modal Pre-training on Plot Table Pairs](https://arxiv.org/abs/2305.18641)
<br/>
[Mingyang Zhou](https://scholar.google.com/citations?user=hIpaL2wAAAAJ&hl=en), [Yi R. Fung](https://yrf1.github.io/), [Long Chen](https://zjuchenlong.github.io/), [Christopher Thomas](https://people.cs.vt.edu/chris/), [Heng Ji](http://blender.cs.illinois.edu/hengji.html), [Shih-Fu Chang](https://www.ee.columbia.edu/~sfchang/)
<br/>
ChartT5 is a vision and language model for chart understanding via pre-training on plot-table pairs. This repository provides the code for pre-training and fine-tuning on the ChartQA Downstream tasks. 

## Setup
Install conda to set up the environment for this code with the following command.
```
conda env create -f chartT5.yml
#activate the virtual environment
conda activate chartT5

# Download T5/BART backbone checkpoint
python download_backbones.py
```
## Download Data and Pre-trained Checkpoints
The data for pre-training and fine-tuning is downloadable via this link:
<br/>
[Preprocessed_Data](https://drive.google.com/file/d/1QZNz6_2fobrVtU4DEzrM0ZZghBlSLRSJ/view?usp=sharing)


The pre-trained checkpoints can be downloaded via this link:
<br/>
[Pre-trained Checkpoint](https://drive.google.com/file/d/1MPps6hMrvVmP_ORjsNeMLd_2YFlhtGPW/view?usp=sharing)

We are still working on preparing the extracted visual features for download. However, you can also extract visual features from the images with the following instructions.
```
conda env create -f feature_extractor.yml
conda activate chart_feature_extractor

cd feature_extraction
#extract features for chartQA dataset
python chartqa_proposal.py --data_root /path/to/your/chartvqa --split train/val/test
```

## Pre-training
After extract the data, change the `pretrain_datadir` in `ChartT5/src/chart_pretrain_data.py` to the /path/to/extracted_data/pretrain . 
Then run the following command to start the pre-training:

```
cd ChartT5
bash scripts/Chartpretrain_VLT5.sh 2
```

## Downstream Task Fine-tuning
### Chart VQA
After extract the data, change the `chartqa_root` in `ChartT5/src/chartqa_data.py` to `/path/to/extracted_data/chart_qa`. Also change the `src_folder` in `ChartT5/scripts/ChartQA_VLT5.sh` to  `/path/to/extracted_data/chart_qa`. 
Then run the following command to start the fine-tuning:
```
cd ChartT5
bash scripts/ChartQA_VLT5.sh 2
```

## Citation
Please cite our paper if you use our model in your works:
```
@inproceedings{zhou2023chartt5,
  title     = {Enhanced Chart Understanding in Vision and Language Task via Cross-modal Pre-training on Plot Table Pairs},
  author    = {Mingyang Zhou, Yi R. Fung, Long Chen, Christopher Thomas, Heng Ji, Shih-Fu Chang},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  year      = {2023}
}
```
## Acknowledge
Our code is mainly based on [VLT5](https://github.com/j-min/VL-T5). We thank the author for opening source their code and checkpoints. Portions of the code also uses the resource from [ChartQA](https://github.com/vis-nlp/ChartQA).

## Liscense
MIT





