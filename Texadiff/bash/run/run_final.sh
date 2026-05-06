

python3 run.py --pretrained_model_name_or_path "checkpoint/stable-diffusion-xl-base-1.0" \
    --unet_model_name_or_path "" \
    --controlnet_model_name_or_path "" \
    --controlnet_scale 1.0 \
    --vae_model_name_or_path "checkpoint/sdxl-vae-fp16-fix" \
    --validation_prompt "high-resolution. 8k, clean" \
    --negative_prompt "blurry, dotted, noise, raster lines, unclear, low-resolution, over-smoothed" \
    --validation_image "" \
    --output_dir "" \
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

