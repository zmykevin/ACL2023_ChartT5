# The name of experiment
name=VLT5

# output=snap/pretrain/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 47769 \
    src/chart_pretrain_inference.py \
        --distributed --multiGPU --fp16 \
        --test val \
        --batch_size 160 \
        --num_workers 1 \
        --losses 'table_lmh,table_lmd' \
        --backbone 't5-base' \
        --output '/dvmm-filer2/projects/mingyang/semafor/chart_pretrain/output/exp_full_pretrain_tablelmhd_ocrposition_ocrcopy/last_random' \
        --load '/dvmm-filer2/projects/mingyang/semafor/chart_pretrain/output/exp_full_pretrain_tablelmhd_ocrposition_ocrcopy/Epoch30' \
        --num_beams 5 \
        --ocr_tags \
        --visfeat_type "chart_element" \
        --ocr_position_encoding "ocr_bbox" \
        --ocr_copy \
        #--tableMaskRate  0.8 \
        #Mask out the whole table and decode the whole table
