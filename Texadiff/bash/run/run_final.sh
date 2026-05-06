

python3 run.py --pretrained_model_name_or_path "checkpoint/stable-diffusion-xl-base-1.0" \
    --unet_model_name_or_path "/home/nju/zez/ADRSSR/train_ckpt/1107v_final/checkpoints/checkpoint-30000" \
    --controlnet_model_name_or_path "/home/nju/zez/ADRSSR/train_ckpt/1107v_final/checkpoints/checkpoint-30000" \
    --controlnet_scale 1.0 \
    --vae_model_name_or_path "checkpoint/sdxl-vae-fp16-fix" \
    --validation_prompt "high-resolution. 8k, clean" \
    --negative_prompt "blurry, dotted, noise, raster lines, unclear, low-resolution, over-smoothed" \
    --validation_image "/home/nju/zez/ADRSSR/testdata/SIRI-WHU/12class_png" \
    --output_dir "infer_cvpr/1107final_siri_maskmodelv4_thr0.35_maxpool_in100to500_min2000_withfix" \
    --variant fp16 \
    --num_inference_steps 20 \
    --use_safetensors \
    --first_stage_model_config "config/psr_rtdm.yaml" \
    --num_validation_images 1 \
    --color_fix \
    --sr_model \
    --guidance_scale 5 \
    --thr 0.35 \
    --min_area 1000 \

