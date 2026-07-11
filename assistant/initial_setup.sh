#!/bin/bash
python3.12 -m venv vinkona_env
source vinkona_env/bin/activate
pip install --upgrade pip
# adapt cu132 to current cuda version as required
pip install torch --index-url https://download.pytorch.org/whl/cu132
# cound number of devices available to the environment (needs to be at least 1)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
pip install -r requirements.txt
# this will fail unless you have a huggingface account:
# hf login .. and then provide your access token
hf download nvidia/personaplex-7b-v1
