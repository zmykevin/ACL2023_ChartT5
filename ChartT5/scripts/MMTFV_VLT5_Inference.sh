# The name of experiment
name=VLT5

output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 47772 \
    src/MMTFV_itm_inference.py \
        --distributed --multiGPU \
        --test test_hard \
        --num_workers 8 \
        --backbone 't5-base' \
        --output '/dvmm-filer2/projects/mingyang/semafor/infographic_itm/output/exp_chartT5_noimage/best_' \
        --load '/dvmm-filer2/projects/mingyang/semafor/infographic_itm/output/exp_chartT5_noimage/BEST' \
        --num_beams 5 \
        --valid_batch_size 64 \
        --src_folder "/dvmm-filer2/projects/mingyang/semafor/infographic_itm/" \
        --raw_label \
        --fp16 \
        --visfeat_type "chart_element" \
        --ocr_tags \
        --classifier \
        # --use_vision 'False' \
        # --ocr_position_encoding "ocr_bbox" \
        #--ocr_copy \
        #--num_modeling \
        #--dataset_name 'plotqa' \
        #--num_modeling \
        # --ocr_copy \
        # --scif_num \
        # --src_folder "/dvmm-filer2/projects/mingyang/semafor/chart_qa/ChartQAv1/" \
