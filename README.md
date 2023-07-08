# ChartT5
The repository for chart-table pre-training, which aims to learn a robust representation of chart image in order to conduct various chart image related v+l downstream tasks. 

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

### Chart Summarization
For bookkeeping sake, you can find processed data files [here](https://uofi.box.com/s/bkox2p00hp3jgzw15hmpushy5lh8b09u), and trained chart summarization checkpoint [here](https://uofi.box.com/s/e9noxyzea7xeppp50pn1sx1lm4wg4xab).

Some hard-coded parts to take care of before running Chart-to-text:
- Download the original VLT5 repo's pretrained checkpoint, "Epoch30.pth", and place under `ChartT5/VL-T5/`
- Place `Chart-to-text/` containing ocr and features etc, downloadable under link above, under `ChartT5/VL-T5/datasets/`
```
cd VLT5

# Note that some of the evaluation metric requires
# library call with dependency conflict. Hence, in current implementation,
# eval is execulated as a subprocess call using a different conda environment.
# - Set up the ChartSumm eval environment through `conda env create -f VL-T5/environment_eval.yml` first.
# - Then, run command below using regular VL-T5 environment

# For training:
python VL-T5/src/ChartSummarization.py --epochs 30 --lr 5e-5 --dataset_name statista \
                                       --src_folder ${PWD}/datasets/Chart-to-text
python VL-T5/src/ChartSummarization.py --epochs 20 --lr 5e-5 --dataset_name pew \
                                       --src_folder ${PWD}/datasets/pew  #Chart-to-text
# For Testing:
python VL-T5/src/ChartSummarization.py --epochs 0 --dataset_name statista --src_folder ${PWD}/datasets/Chart-to-text
python VL-T5/src/ChartSummarization.py --epochs 0 --dataset_name pew --src_folder ${PWD}/datasets/Chart-to-text
```

To use the adapted feautures,
you can also enable it via adding the flag:
```
--visfeat_type "chart_element" \
```

To use the ocr position encoding, youo can enable it via adding the following flag:
```
--ocr_position_encoding "ocr_bbox" \
--ocr_tags \
```
## Results

Started training from a VLT5 model, with a general purpose pretrained T5 decoder.

| VLT5      | BLEU | CS | BLEURT | CIDER | PPL
| ----------- | ----------- | -- | --- |--- |--- 
| Statista      |  31.4     | 35.9 | -0.345 | 0.90 | 14.8
| Pew | 20.3 | 42.65 | -0.43 | 0.20 | 7.26

| Best Baseline    | BLEU | BLEURT | CIDER | PPL
| ----------- | ----------- | --- |--- |--- 
| Statista      |  35.29     | 0.10 | 4.43 | 8.59
| Pew   | 10.49 | -0.35 | 2.20| 10.11

