# The name of experiment
name=VLT5

# output=snap/pretrain/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 47775 \
    src/chart_pretrain.py \
        --distributed --multiGPU --fp16 \
        --train train\
        --valid val \
        --batch_size 24 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --losses 'table_lmh,table_lmd' \
        --backbone 't5-base' \
        --output '/dvmm-filer2/projects/mingyang/semafor/chart_pretrain/output/exp_pretrain' \
        --epoch 30 \
        --ocr_tags \
        --visfeat_type "chart_element" \
        --ocr_position_encoding "ocr_bbox" \
        --ocr_copy \
        
        

        