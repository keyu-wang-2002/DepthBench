python3 -m venv depthbench

source depthbench/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118

cd /DepthBench/pre-train/OLMo-core

python -m pip install -e ".[wandb,transformers]"
python -m pip install datasets pyarrow

