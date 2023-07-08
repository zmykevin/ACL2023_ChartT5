# The name of experiment
name=VLT5

# output=snap/nlvr/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 47772 \
    src/MMTFV_itm.py \
        --distributed --multiGPU \
        --train train \
        --valid val \
        --test test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 10 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output '/dvmm-filer2/projects/mingyang/semafor/infographic_itm/output/exp_chartT5_noevidence' \
        --batch_size 8 \
        --visfeat_type "chart_element" \
        --classifier \
        --use_article False \
        --use_vision False \
        #--ocr_tags \
        # --load snap/pretrain/VLT5/Epoch30 \
        # --max_text_length 40 \
        # --output '/dvmm-filer2/projects/mingyang/semafor/chart_itm/output/exp_test' \