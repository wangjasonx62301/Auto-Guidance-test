# Auto-Guidance-test
auto guidance test on image diffusion

```bash
python test.py \
--guidance_mode autog \
--guidance_scale 0.3 \
--epoch 500 \
--ckpt_path /home/jasonx62301/for_python/Auto-Guidance-test/runs/ddpm/ckpt_epoch_250.pt \
--train_with_autog True \
--logdir runs/autog03

```