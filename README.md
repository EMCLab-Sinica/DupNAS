# 🚀 DupNAS: Splitting Bottlenecks: Memory-Aware Neural Architecture Search for Multi-Branch TinyML

> **Official artifact repository for DupNAS**  
> Memory-constrained neural architecture search, tensor splitting, and deployment for TinyML on MCU-class devices.


<!-- [![Paper](#)](#) [![License](#)](#) [![Python](#)](#) [![Platform](#)](#) -->

---
## 📝 Overview
---
<!-- This repository contains the full artifact for reproducing the NAS, model splitting, fine-tuning, ONNX export, and MCU deployment workflow used in DupNAS. -->

---
## 📌 Directory/File Structure
---
Below is an explanation of the key directories/files found in this repository.

<!-- TiNAS/NASBase/ss_optimization contains the implementation for the search space optimizer (adapted from TinyNAS)
TiNAS/NASBase/evo_search contains the implementation for the evolutionary search strategy (adapted from TinyNAS)
TiNAS/NASBase/hw_cost contains the implementation for the intermittent inference cost model and intermittent execution design explorer (adapted from iNAS)
TiNAS/NASBase/model contains the search space definition, supernet and subnet structure
TiNAS/tools/imo_sensitivity contains the implementation for the IMO sensitivity analysis tool
TiNAS/DNNDumper is a helper module used to convert the derived solutions into a custom C data structure recognizable by the intermittent inference runtime library
TiNAS/settings contains the settings used for evaluation (for different datasets and baseline approaches)
TiNAS/settings.py contains the overall NAS settings and implementation for managing/loading settings files
TiNAS/misc_scripts contains miscellaneous helper scripts
TiNAS/requirements.txt contains the dependencies required to run TiNAS
intermittent-inference-library contains the intermittent inference runtime library developed for the TI-MSP430FR5994 (extended from iNAS's inference library) -->

---
## 🧭 Getting Started
---

### 💡 Prerequisites

- `Python 3.9` is recommended.
- Install the required Python packages listed in `requirements.txt` with:
  `python3.9 -m pip install -r requirements.txt`
- [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install/overview) is optional, but recommended for managing Python environments.
- The main dataset used in this project is [ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100/data). You can prepare and load it using: `\TiNAS\NASBase\load_image100.py`.
- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)
- [STM32F746NG MCU](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)

### 🔧Setup and Build for DupNAS

1. Download/clone this repository
2. Create and activate a Python environment, and install dependencies
3. Prepare the ImageNet-100 dataset and update dataset paths.
4. Run the NAS pipeline:
  ```python
  python3.9 -m NASBase.run_nas --stages <stage> --arc <arc> --dataset IMAGE100 --mode <mode> --vmsize <vmsize> --suffix <suffix> --no-rlogger
  ```
  
  ### 📝 NAS Command Arguments
  | Option | Description | Candidate Values |
  |---|---|---|
  | `--stages` | Number of NAS stages: ssopt, training, evosearch, fine-tuning | `1`, `2`, `3`, `4` |
  | `--arc` | Backbone architecture | `mbv2`, `shuffle`, `incept` |
  | `--dataset` | Dataset used for search | `IMAGE100` |
  | `--mode` | TS optimization  | `pdq`, `tinyts`, `tinynas`, `none` |
  | `--vmsize` | VM constraint in KB | `96`, `128`, `256` |
  | `--suffix` | Experiment suffix for output naming | user-defined string |
  
5. catch the solution to fill `spec_model.txt`
6. generate the onnx for solution networks by `\TiNAS\NASBase\spec_onnx_gen.py`

### ✂️ Model Splitting


### ⚙️ Inference

---


---
## 🧩 Results
---

### Accuracy

| Model | TS Mode | VM = 96 KB | VM = 128 KB | VM = 256 KB |
|---|---|---:|---:|---:|
| MobileNet   | DupNAS  | XX | XX | XX |
|             | TinyTS  | XX | XX | XX |
|             | PatchTS | XX | XX | XX |
| ShuffleNet  | DupNAS  | XX | XX | XX |
|             | TinyTS  | XX | XX | XX |
|             | PatchTS | XX | XX | XX |
| InceptionNet | DupNAS  | XX | XX | XX |
|             | TinyTS  | XX | XX | XX |
|             | PatchTS | XX | XX | XX |



<!-- ---
### Included in this repository

#### 🔍 NAS-related
- Full DupNAS source code
- Exact dependency list (`requirements.txt`)
- Architecture search space parameters and value ranges
- Trained supernet checkpoints (`.pth`)
- Fine-tuned subnet solutions
- Architecture definitions, ONNX / PTH files, and summary metadata
- Estimated peak VM / NVM usage and latency proxy
- Example split models and ONNX visualizations
- Training and fine-tuning settings
- ImageNet-100 preparation details

#### 🔧 Deployment-related
- TFLite Micro model converter code
- TFLite Micro runtime reference / patches
- STM32 compile and run dependencies
- Real inference latency on deployed models
- CSV files for all figure data


## 🔍 NAS Pipeline

This section describes the full NAS workflow included in this repository.

### Pipeline Stages
1. Supernet training
2. Architecture search under memory constraints
3. Model splitting / duplication
4. Fine-tuning selected subnet solutions
5. ONNX export
6. Memory / latency estimation

### Included Components
- Supernet training code
- NAS search code
- Search space definition
- Fine-tuning scripts
- ONNX export scripts
- Result analysis utilities

### Main Scripts

| Task | Script / Path | Description |
|------|---------------|-------------|
| Supernet training | `src/...` | Train the supernet |
| NAS search | `src/...` | Run DupNAS search |
| Fine-tuning | `src/...` | Fine-tune selected subnets |
| ONNX export | `src/...` | Export final subnet models |
| Evaluation | `src/...` | Summarize results |

---

## 🧾 Architecture Search Space

> Add the full search-space table here, since it was removed from the paper.

| Parameter | Description | Values |
|---|---|---|
| Backbone | Model family |  |
| Input resolution | Input image size |  |
| Width multiplier | Channel scaling |  |
| Kernel size | Conv kernel choices |  |
| Expansion ratio | Bottleneck expansion |  |
| Depth / stages | Number of blocks |  |
| Duplication factor | Splitting level |  |
| VM constraint | SRAM budget |  |

---


---

## 🧩 Fine-Tuned Subnet Solutions

| Model | Subnet | Accuracy | Peak VM | Peak NVM | Latency | Files |
|---|---|---:|---:|---:|---:|---|
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

### Suggested metadata file
- `subnet_solutions/summary.csv`

---

## 🖼️ Split Model Examples

### Original vs. Split
Add example ONNX models and graph visualizations here.

<p align="center">
  <img src="figures/example_split.png" width="85%">
</p>

**Figure:** Add a caption describing the original and split model comparison.

---

## 🏋️ Training / Fine-Tuning Settings

### Supernet Training

| Setting | Value |
|---|---|
| Dataset |  |
| Batch size |  |
| Optimizer |  |
| Learning rate |  |
| Epochs |  |

### Subnet Fine-Tuning

| Setting | Value |
|---|---|
| Batch size |  |
| Optimizer |  |
| Learning rate |  |
| Epochs |  |
| Initialization |  |

---

## 🗃️ Dataset Preparation

### Dataset
- Dataset: `ImageNet-100`
- Source: original ImageNet
- Usage: supernet training, subnet fine-tuning, evaluation

### Preparation
Describe:
- how classes were selected
- how train / validation splits were generated
- whether data were filtered or modified
- preprocessing and augmentation

### Reproducibility

```bash
python datasets/imagenet100/prepare_imagenet100.py \
  --src <path-to-imagenet> \
  --dst datasets/imagenet100/
```

---

## 🔧 Deployment Pipeline

### Pipeline Stages
1. Convert trained subnet models
2. Prepare TFLite Micro model files
3. Integrate models into STM32 project
4. Compile firmware
5. Run inference on device
6. Collect real latency results

### Included Components
- TFLite Micro converter code
- STM32 deployment project
- Runtime configuration files
- Real-device latency results
- Deployment CSV summaries

### Main Scripts / Paths

| Task | Script / Path | Description |
|---|---|---|
| Model conversion | `deployment/tflm_converter/...` | Convert exported subnet model |
| STM32 project | `deployment/stm32/...` | Deployment project files |
| Latency results | `deployment/latency_results/...` | Real-device measurements |

### Dependencies
- TFLite Micro version: `...`
- STM32CubeIDE: `...`
- ARM GCC: `...`
- CMSIS / BSP: `...`

---

## ⏱️ Deployment Results

| Model | Device | Real Latency (ms) | Notes |
|---|---|---:|---|
|  |  |  |  |
|  |  |  |  |

---

## 📊 CSV Files for Figures

| Figure | CSV File | Description |
|---|---|---|
|  |  |  |
|  |  |  |

---

## 🔁 Reproducibility

### Suggested Order
1. Install dependencies
2. Prepare dataset
3. Run NAS / load checkpoints
4. Fine-tune subnet
5. Export ONNX
6. Convert for deployment
7. Run on STM32
8. Compare with reported CSV results

---

## ✅ Artifact Checklist

### NAS-related
- [ ] Full DupNAS source code
- [ ] `requirements.txt` with exact package versions
- [ ] Search space table
- [ ] Trained supernet checkpoints
- [ ] Fine-tuned subnet solutions
- [ ] Architecture definitions / ONNX / PTH files
- [ ] Accuracy / VM / NVM / latency metadata
- [ ] Split-model examples and visualizations
- [ ] Training / fine-tuning settings
- [ ] Dataset preparation details

### Deployment-related
- [ ] TFLite Micro converter code
- [ ] TFLite Micro dependency list
- [ ] TFLite Micro engine link or patch
- [ ] STM32 compile / run dependencies
- [ ] Real-device latency results
- [ ] CSV files for all figures

---

## 📎 Citation

```bibtex
@inproceedings{dupnas,
  title     = {Title of Your Paper},
  author    = {Author 1 and Author 2 and Author 3},
  booktitle = {Conference Name},
  year      = {2026}
}
```

---

## 📜 License

Add license information here.

---

## 🙋 Contact

- Name:
- Email:

 -->