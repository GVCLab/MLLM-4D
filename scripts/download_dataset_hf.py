from huggingface_hub import snapshot_download

def download_datasetl():
    snapshot_download(
        repo_id="flow666/MLLM-4D-Datasets", 
        repo_type="dataset",
        local_dir="../data/MLLM-4D-Datasets",
        local_dir_use_symlinks=False,
    )

download_dataset()