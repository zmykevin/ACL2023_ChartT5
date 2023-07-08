name=VLT5

# output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 47775 \
    src/chartqa.py \
    --distributed --multiGPU \
    --train train \
    --valid val \
    --test test \
    --optim adamw \
    --warmup_ratio 0.05 \
    --clip_grad_norm 5 \
    --lr 2e-4 \
    --epochs 40 \
    --num_workers 16 \
    --backbone 't5-base' \
    --output '/dvmm-filer2/projects/mingyang/semafor/chart_qa/output/exp_no_pretrain_num_modeling' \
    --num_beams 5 \
    --batch_size 12 \
    --valid_batch_size 64 \
    --src_folder "/dvmm-filer2/projects/mingyang/semafor/chart_qa/ChartQAv1/" \
    --raw_label \
    --fp16 \
    --visfeat_type "chart_element" \
    --ocr_tags \
    --ocr_position_encoding "ocr_bbox" \
    --num_modeling \
    #--ocr_copy \
    #--load "/dvmm-filer2/projects/mingyang/semafor/chart_pretrain/output/exp_synthetic_pretrain_ocrcopy_fix/Epoch30"
    # --load "/dvmm-filer2/projects/mingyang/semafor/chart_pretrain/output/exp_synthetic_pretrain_ocrcopy_special/Epoch30"
    #--num_modeling \
    # --ocr_copy \
    # --load "/dvmm-filer2/projects/mingyang/semafor/chart_pretrain/output/exp_synthetic_pretrain_ocrcopy_special/Epoch30"
    
    #
    #--load "/dvmm-filer2/projects/mingyang/semafor/chart_pretrain/output/exp_synthetic_pretrain_tableonly_ocrcopy_tableMaskrate0.15/Epoch30"
    #--num_modeling \

    
    #--ocr_copy \
    #--load "/dvmm-filer2/projects/mingyang/semafor/chart_pretrain/output/exp_synthetic_pretrain_tableonly_ocrcopy_tableMaskrate0.15/Epoch30"
    #--scif_num \
    # --ocr_copy \
    #--load "/dvmm-filer2/projects/mingyang/semafor/chart_pretrain/output/exp_full_pretrain_tablelmhd_ocrposition_ocrcopy/Epoch30"  \

    
    
    