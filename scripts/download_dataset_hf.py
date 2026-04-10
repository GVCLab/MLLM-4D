from huggingface_hub import snapshot_download

def download_dataset():
    snapshot_download(
        repo_id="flow666/MLLM-4D-Datasets", 
        repo_type="dataset",
        local_dir="../data/MLLM-4D-Datasets",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="shijiezhou/VLM4D", 
        repo_type="dataset",
        local_dir="../data/VLM4D",
        local_dir_use_symlinks=False,
    )

download_dataset()