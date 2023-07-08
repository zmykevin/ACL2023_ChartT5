# The name of experiment
name=VLT5

# output=snap/nlvr/$name

PYTHONPATH=$PYTHONPATH:./src \
python src/ChartSummarization.py \
        --epochs 20 \
        --num_workers 1 \
        --backbone 't5-base' \
        --output '/dvmm-filer2/projects/mingyang/semafor/chart_summary/output/exp_debug' \
        --oscar_tags \
        --dataset_name "statista" \
        --batch_size 4 \
        --num_beams 4 \
        --visfeat_type "chart_element" \
        --ocr_position_encoding "ocr_bbox" \
        --ocr_tags \
        --ocr_copy \
# python -m torch.distributed.launch \
#     --nproc_per_node=$1 \
#     --master_port 47771 \
#     src/ChartSummarization.py \
#         --distributed --multiGPU \
#         --epochs 20 \
#         --num_workers 1 \
#         --backbone 't5-base' \
#         --output '/dvmm-filer2/projects/mingyang/semafor/chart_summary/output/exp_test' \
#         --oscar_tags \
#         --visfeat_type "chart_element" \
#         --dataset_name "statista" \
#         --batch_size 32 \
#         --num_beams=4
        # --train train \
        # --valid val \
        # --test val\
        # --optim adamw \
        # --warmup_ratio 0.1 \
        # --clip_grad_norm 5 \
        # --lr 1e-4 \
        # --load snap/pretrain/VLT5/Epoch30 \
        # --max_text_length 40 \
