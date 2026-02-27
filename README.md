### MLLM-4D: Towards Visual-based Spatial-Temporal Intelligence

Xingyilang Yin, Chengzhengxu Li, Jiahao Chang, Chi-Man Pun, Xiaodong Cun

[ArXiv]() | [PDF]() | [Model](https://huggingface.co/flow666/MLLM-4D/tree/main)

🤗 If you find MLLM-4D useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

TL;DR: We propose MLLM-4D, a framework for advancing visual-based spatial-temporal intelligence. Our method is motivated by human innate cognition: humans receive 2D visual inputs, while the brain's inherent spatiotemporal intelligence enables understanding and reasoning of relationships within the 4D world.


![Teaser](assets/mllm4d_teaser.png)

## ⚙️ Setup

### 1. Clone MLLM-4D
```bash
git clone https://github.com/GVCLab/MLLM-4D.git
cd MLLM-4D
```
### 2. Setup environments
MLLM-4D is tested with CUDA 12.1/12.8 on H100.
```bash
conda create -n mllm4d python=3.10
conda activate mllm4d 
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

```

### 3. Download pretrained models
```bash
python scripts/download_ckpt_hf.py
```

<!-- ### 4. Download the datasets
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

## 📋 TODO
- [ ] We have completed the code and data clean up. (Release coming soon!)
- [ ] RFT Stage: Release the `MLLM4D-R1-30k` dataset and `Reinforcement Fine-Tuning code`!
- [ ] Cold-Start Phase: Release the `Cold-Start Data` and `Cold-Start Fine-Tuning code`!
- [ ] SFT Stage: Release the `MLLM4D-2M` dataset and `Supervised Fine-Tuning code`!
- [x] **[2026.02.28]** 🔥 Release the `arXiv paper`, `inference demo`, and `pretrained weights`!

## 🤗 Acknowledgement
Our work is built upon [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), thanks to their invaluable contributions

## 📜 Citation
If you find the work useful, please consider citing:
```BibTeXw

```
