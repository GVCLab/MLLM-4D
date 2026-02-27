import os
from huggingface_hub import snapshot_download

def download_model():
    snapshot_download(
        repo_id="flow666/MLLM-4D",
        local_dir="../checkpoints/MLLM-4D",
        local_dir_use_symlinks=False,
    )


download_model()