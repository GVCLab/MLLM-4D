## ✨MLLM-4D: Towards Visual-based Spatial-Temporal Intelligence✨

#### [Xingyilang Yin*](https://flow0314.github.io/)<sup>1,2</sup>, [Chengzhengxu Li*](https://scholar.google.com/citations?user=NSWsjzcAAAAJ&hl=zh-CN)<sup>3</sup>, [Jiahao Chang](https://github.com/Jiahao620)<sup>4</sup>, [Chi-Man Pun](https://cmpun.github.io/)<sup>1,📫</sup>, [Xiaodong Cun](https://vinthony.github.io/academic/)<sup>2,📫</sup>

[ArXiv](https://arxiv.org/abs/2603.00515) | [PDF](https://arxiv.org/pdf/2603.00515) | [Model](https://huggingface.co/flow666/MLLM-4D/tree/main) | Dataset

###### <sup>1</sup> University of Macau, <sup>2</sup> GVC Lab, Great Bay University, <sup>3</sup> Xi’an Jiaotong University, <sup>4</sup> CUHKSZ

>TL;DR: MLLM-4D achieves advanced visual-based spatial-temporal intelligence. Our method specifically focuses on understanding and reasoning about the time-evolving relationships between objects and camera within 3D space. Read our paper for more details.

![Teaser](assets/mllm4d_teaser.png)

## ⚙️ Setup

### 1. Clone MLLM-4D
```bash
git clone https://github.com/GVCLab/MLLM-4D.git
cd MLLM-4D
```
### 2. Setup Environments
MLLM-4D is tested with CUDA 12.1/12.8 on H100.
```bash
conda create -n mllm4d python=3.10
conda activate mllm4d 
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 3. Download Pretrained Models
```bash
python scripts/download_ckpt_hf.py
```

<!-- ### 4. Download the Datasets
```bash
``` -->

## 💫 Inference 
### 1. Inference Demo
```bash
# for MLLM-4D-SFT
python scripts/inference.py --model_type "MLLM-4D-SFT" --model_path PATH-to-MLLM-4D-SFT
# for MLLM-4D-RFT
python scripts/inference.py --model_type "MLLM-4D-RFT" --model_path PATH-to-MLLM-4D-RFT
```

## 🚂 Training
### 1. Supervised Fine-Tuning Using Our MLLM4D-2M Dataset.
Please set up the parameters in [\_\_init\_\_.py](./qwen-vl-sft/qwenvl/data/__init__.py) and [sft_qwen3-vl-8b_mllm4d-2M.sh](./qwen-vl-sft/scripts/sft_qwen3-vl-8b_mllm4d-2M.sh).
```bash
cd qwen-vl-sft
bash scripts/sft_qwen3-vl-8b_mllm4d-2M.sh
```

### 2. Cold-Start Fine-Tuning Using Our Cold-Start Data.
Please set up the parameters in [\_\_init\_\_.py](./qwen-vl-sft/qwenvl/data/__init__.py) and [sft_mllm-4d-sft_cold-start.sh](./qwen-vl-sft/scripts/sft_mllm-4d-sft_cold-start.sh).
```bash
cd qwen-vl-sft
bash scripts/sft_mllm-4d-sft_cold-start.sh
```

### 3. Reinforcement Fine-Tuning Using Our MLLM4D-R1-30k Dataset.


## 📋 TODO
- [ ] We have completed the code and data cleanup. Release coming soon!
- [ ] RFT Stage: Release the `MLLM4D-R1-30k` dataset and `Reinforcement Fine-Tuning code`!
- [ ] Cold-Start Phase: Release the `Cold-Start Data` and `Cold-Start Fine-Tuning code`!
- [ ] SFT Stage: Release the `MLLM4D-2M` dataset and `Supervised Fine-Tuning code`!
- [x] **[2026.02.28]** 🔥 Release the `arXiv paper`, `inference demo`, and `pretrained weights`!

## 🤗 Acknowledgement
If you find MLLM-4D useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

Our work is built upon [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), thanks to their invaluable contributions!

## 📚 Citation
If you find the work useful, please consider citing:
```BibTeXw
@article{yin2026mllm4d,
    title={MLLM-4D: Towards Visual-based Spatial-Temporal Intelligence},
    author={Yin, Xingyilang and Li, Chengzhengxu and Chang, Jiahao and Pun, Chi-Man and Cun, Xiaodong},
    journal={arXiv preprint arXiv:2603.00515},
    year={2026}
}
```
