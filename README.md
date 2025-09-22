# Auto-Guidance-test
auto guidance test on image diffusion

```bash
CUDA_VISIBLE_DEVICES=2 \
python test.py \
--guidance_mode autog \
--guidance_scale 0.5 \
--epoch 500 \
--ckpt_path /home/jasonx62301/for_python/Auto-Guidance-test/runs/autog/ckpt_epoch_750.pt \
--train_with_autog True \
--logdir runs/autog_with_selfg_ttt \
--bad_model_ckpt /home/jasonx62301/for_python/Auto-Guidance-test/runs/ddpm/ckpt_epoch_250.pt
```