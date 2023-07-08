name=VLT5

# output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 47773 \
    src/chartqa.py \
    --distributed --multiGPU \
    --train train \
    --valid testv1 \
    --test testv2 \
    --optim adamw \
    --warmup_ratio 0.05 \
    --clip_grad_norm 5 \
    --lr 2e-4 \
    --epochs 2 \
    --num_workers 16 \
    --backbone 't5-base' \
    --output '/dvmm-filer2/projects/mingyang/semafor/chart_qa/output/exp_plotqa_no_pretrain' \
    --num_beams 5 \
    --batch_size 18 \
    --valid_batch_size 64 \
    --src_folder "/dvmm-filer2/projects/mingyang/semafor/chart_qa/PlotQA/" \
    --raw_label \
    --fp16 \
    --visfeat_type "chart_element" \
    --ocr_tags \
    --ocr_position_encoding "ocr_bbox" \
    --ocr_copy \
    --dataset_name 'plotqa' \