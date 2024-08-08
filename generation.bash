conda create -n udit python==3.10
conda activate udit
pip install -r requirements.txt
wget https://huggingface.co/yuchuantian/U-DiT/resolve/main/U-DiT-L-1000k.pt
python generation_single_gpu.py --ckpt=U-DiT-L-1000k.pt --image-size=256 --model=U-DiT-L  --cfg-scale=1.5
