# ChartT5
[Enhanced Chart Understanding in Vision and Language Task via Cross-modal Pre-training on Plot Table Pairs](https://arxiv.org/abs/2305.18641)
<br/>
[Mingyang Zhou](https://scholar.google.com/citations?user=hIpaL2wAAAAJ&hl=en), [Yi R. Fung](https://yrf1.github.io/), [Long Chen](https://zjuchenlong.github.io/), [Christopher Thomas](https://people.cs.vt.edu/chris/), [Heng Ji](http://blender.cs.illinois.edu/hengji.html), [Shih-Fu Chang](https://www.ee.columbia.edu/~sfchang/)
<br/>
ChartT5 is a vision and language model for chart understanding via pre-training on plot-table pairs. This repository provides the code for pre-training and fine-tuning on the ChartQA Downstream tasks. 

## Env Set Up
The main code is built on top of [VL-T5](https://github.com/j-min/VL-T5), follow there env set up instructions. 
```
cd VLT5
#then run the set up commands in VL-T5 repo

# Next, NEW (from Yi): also install the summarization task eval environment
# Due to version conflicts, eval part will be called as a subprocess run
# using this separate virtual environment. Everything is streamlined tho.
conda env create -f VL-T5/environment_eval.yml
```

## Pre-training
Download the Preprocecssed Chart Summary Dataset [Here](https://drive.google.com/file/d/1mXXLtbqHkGlPre7hyfMwKkE7mcutl3RG/view?usp=sharing)
After extract the data, change the datapath in the chart_pretrain_data.py accordingly. 
Then run the following command:

```
cd VLT5/VLT5

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/chart_pretrain.py \
        --distributed --multiGPU --fp16 \
        --train chartsum_train\
        --valid chartsum_val \
        --batch_size 80 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --losses 'lm,itm' \
        --backbone 't5-base' \
        --output '/path/to/save/checkpoints' \
        --epoch 30 
```

## Downstream Task Fine-tuning
### Chart VQA
Download the Preprrocessed Chart QA Dataset [Here](https://drive.google.com/file/d/1l4qX3XV0Z8b-XQvEHvtUbt044FP4enLX/view?usp=sharing)
After extrarct the data, change the path in the chartqa_data.py accordingly,
Then run the following command
```
cd VLT5/VLT5

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/chartqa.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 15 \
        --num_workers 16 \
        --backbone 't5-base' \
        --output '/path/2/output' \
        --num_beams 5 \
        --batch_size 32 \
        --valid_batch_size 64 \
        --src_folder "/path/2/chartqa/data" \
        --raw_label \
        --fp16 \
```
To run the ChartQA Fine-tuning with OCRs and Adapted Visual Features for Chart Image, pleasea add the following two flags to the training command:
```
--visfeat_type "chart_element" \
--ocr_tags \
```

### Results
| Model      | Val | Test|
| ----------- | ----------- | --- |
| Best Baseline| 42.60 | **45.52** |
| VLT5      |  6.82     | 5.92 |
| ChartT5   | 43.96 | 43.96| 
| ChartT5 (Synthetic Datata Pretrained)| 43.91|44.84|
| ChartT5 (Synthetic+Real Datata Pretrained)| **44.95**|44.64|

### Setup Chart VQA Fine-tuning on Other Datasets. 
Download the ChartQA dataset you will get a folder with the following structure:
```
Chart VQA/
|    +-- features/
|        +-- train_boxes_36.h5
|        +-- val_boxes_36.h5
|        +-- test_boxes_36.h5
|    +-- train/
|        +-- png/
|            +-- img1.png
|            +-- ...
|        +-- ocr_results/
|            +-- img1.json
|            +-- ...
|        +-- data.csv
|    +-- val/
|        +-- png/
|            +-- img1.png
|            +-- ...
|        +-- ocr_results/
|            +-- img1.json
|            +-- ...
|        +-- data.csv
|    +-- test/
|        +-- png/
|            +-- img1.png
|            +-- ...
|        +-- ocr_results/
|            +-- img1.json
|            +-- ...
|        +-- data.csv
```
You can follow this structure to prepare the New VQA dataset (e.g [PlotQA](https://github.com/NiteshMethani/PlotQA))
First create the three **split** folders under the path you want to store your chart vqa dataset, then copy the corresponding chart images into the **png** folder of the three **split** folders. 

To generate the visual features, run the following command:
```
cd VL-T5/feature_extraction
python chartqa_proposal.py --data_root /path/to/your/chartvqa --split train/val/test
```
To extract the ocrs, we use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
Finally, under each **split** folder, you can create the corresponding annotation following the data.csv forrmat. 

Once your data is ready, you can simply modify the line that defines `chartqa_dir` in `chartqa_data.py` to be the path where you create your new dataset. Then you can run the fine-tuning command to fine-tune on this dataset. 




