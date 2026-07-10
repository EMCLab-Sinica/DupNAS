# 🚀 DupNAS: Splitting Bottlenecks: Memory-Aware Neural Architecture Search for Multi-Branch TinyML

---
## 📝 Overview

This project develops DupNAS, a framework that integrates neural architecture search and multi-branch tensor splitting to find high-accuracy networks that are deployable on severely memory-constrained tiny MCU-class devices. To enable this integration, DupNAS shrinks the vast splitting configuration space of multi-branch networks into a smaller set of memory-optimized configurations and explores them in a lightweight manner to resolve memory bottlenecks.


DupNAS is implemented in PyTorch on top of the popular [TinyNAS](https://github.com/mit-han-lab/mcunet) framework for MCUs, with extensions to support multi-branch networks and integration with our splitting strategy. The resulting network solutions are deployed on an [STM32F746](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html) microcontroller (ARM Cortex-M7 CPU, 216 MHz, 320 KB VM, and 1 MB NVM) running the TensorFlow Lite Micro (TFLite-Micro) inference engine. We evaluate DupNAS on three vision-based TinyML network families, namely MobileNetV2, ShuffleNetV2, and InceptionV3, under different memory constraints. All networks are trained on the ImageNet-100 dataset. DupNAS is compared against two existing splitting strategies, TinyTS and PatchTS.



<p align="center">
  <img src="assets/figures/NAS_with_TS.svg" alt="NAS_with_TS" width="500">
</p>
<p align="center">
  <em>Overview of the DupNAS framework</em>
</p>
<!-- This repository contains the full artifact for reproducing the NAS, model splitting, fine-tuning, ONNX export, and MCU deployment workflow used in DupNAS. -->



## 📌 File Structure

Below is a brief description of the main directories and files in this repository.

- `/DupNAS/NASBase/duplication` implements the DupNAS splitting configuration exploration algorithm, which is invoked during the NAS process.
- `/DupNAS/NASBase/ss_optimization` contains the NAS architecture space optimization component, adapted from TinyNAS.
- `/DupNAS/NASBase/evo_search` contains the NAS evolutionary search component, adapted from TinyNAS.
- `/DupNAS/NASBase/model` defines the architecture search space, including supernet and subnet architecture definitions.
- `/DupNAS/settings` provides the evaluation settings for different datasets and baselines.
- `/DupNAS/settings.py` defines the global NAS settings.
- `/Inference/Model-converter/` provides the model converter, which applies the splitting configuration to a network solution to generate a split network that can be deployed on TFLite-Micro.  
- `/Inference/Tflm-engine/` provides the build process for TFLite-Micro.
- `/assets/DupNAS_paper_data.xlsx` contains the experimental results presented in the paper.
- `/assets/models/` contains the models used for evaluation and deployment.
- 
---
## 🧭 Getting Started


### 💡 Prerequisites

- `Python 3.9`.
- Install the required Python packages listed in `requirements.txt` with:
  `python3.9 -m pip install -r requirements.txt`
- [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install/overview) (recommended for managing Python environments).
- [ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100/data) dataset. Load the dataset from: `https://www.kaggle.com/datasets/ambityga/imagenet100`.
- [STM32F746NG MCU](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html) deployment device.
- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) development tools for the STM32.
- [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro) inference engine.

### 🔧 Setup and running DupNAS

1. Download or clone this repository.
2. Create and activate a Python environment, then install the required dependencies.
   ```python
   python3.9 -m pip install -r requirements.txt
   ```
3. Configure access to the ImageNet-100 dataset
   * Open the [ImageNet-100 Kaggle page](https://www.kaggle.com/datasets/ambityga/imagenet100).
   * Generate a Kaggle API token.
   * Export the token:
     ```bash
     export KAGGLE_API_TOKEN="<your-token>"
     ```
   * Store the token in the Kaggle configuration directory:
     ```bash
     mkdir -p ~/.kaggle
     echo "<your-token>" > ~/.kaggle/access_token
     chmod 600 ~/.kaggle/access_token
     ```
   The dataset is downloaded and prepared automatically when DupNAS is run for the first time stage 2.


  - go to https://www.kaggle.com/datasets/ambityga/imagenet100
  - generage a Kaggle api token
  - `export KAGGLE_API_TOKEN= <Your token>`
  - ` mkdir -p ~/.kaggle && echo <Your token> > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`
4. Copy the configuration file for the target architecture and run DupNAS:
   ```bash
   cp settings/settings-<arc>.py settings.py

   python3.9 -m NASBase.run_nas \
     --stages <stage> \
     --arc <arc> \
     --dataset IMAGE100 \
     --mode <mode> \
     --vmsize <vmsize> \
     --suffix <suffix> \
     --no-rlogger
   ```
  
  📝 Arguments
  | Option | Description | Candidate Values |
  |---|---|---|
  | `--stages` | Number of NAS stages: ssopt, training, evosearch, fine-tuning | `1`, `2`, `3`, `4` |
  | `--arc` | network architecture family | `mbv2`, `shuffle`, `incept` |
  | `--dataset` | Dataset used for search | `IMAGE100` |
  | `--mode` | splitting strategy  | `dupnas`, `tinyts`, `patchts`, `nots` | 
  | `--vmsize` | VM constraint in KB | `96`, `128`, `256` |
  | `--suffix` | Experiment suffix for naming outputs. Use the same suffix for all four stages. | user-defined string |
  
5. After completing Stages 1–4, extract the selected network configuration, `"subnet_choice_per_blk"`, from: `<suffix>_best_solution.json`, and copy the configuration into: `/DupNAS/NASBase/spec_model_<arc>.txt`.
6. For easier use, ONNX generation and tensor-splitting conversion are integrated into the model-converter workflow. Continue with Step 2 in the next section.


### ✂️ Setup and running the model converter

1. Make sure `/DupNAS/NASBase/spec_model_<arc>.txt` has been completed.
2. Go to `/DupNAS/`, then run the corresponding script to generate the selected ONNX models: 
  ```bash
  bash gen_selected_shuffle.sh
  bash gen_selected_mbv2.sh
  bash gen_selected_incept.sh
  ```
3. The generated models are saved in the following locations:
   * Original selected ONNX models:
     ```text
     /DupNAS/genonnx/<arc>/
     ```
   * Tensor-split ONNX models:
     ```text
     /Inference/Model-converter/ts_converted/<arc>/
     ```
4. Go to the `/Inference/onnx-to-tflite/` directory and run the corresponding conversion command:
   ```bash
   bash convert.sh shuffle
   bash convert.sh mbv2
   bash convert.sh incept
   ```
   The conversion uses [onnx2tf](https://github.com/PINTO0309/onnx2tf) to generate integer-quantized TensorFlow Lite Micro models, such as:
   ```text
   xxx_full_integer_quant.tflite
   ```
   The generated TFLite models are saved under `/Inference/onnx-to-tflite/outputs/`
   
### ⚙️ Setup and building TFLite-Micro

1. Copy the converted TFLite-Micro model (`xxx_full_integer_quant.tflite`) into `Tflm-engine/src/models`.
2. Follow [Tflm-engine/README.md](Inference/Tflm-engine/README.md) to build the TFLite-Micro static library (`libtensorflow-microlite.a`).
3. Add the generated static library to your STM32CubeIDE project settings. Then include `Tflm-engine/src/tflm_main.h` and call `tflm_main_xxx` to run inference for the target model.

For more information, please refer to [Tflm-engine/README.md](Inference/Tflm-engine/README.md).


<!-- sudo docker run --rm -it -v $(pwd):/workdir -w /workdir ghcr.io/pinto0309/onnx2tf:1.28.5   -->

---
## 🧩 Evaluation

### Search Space Configuration

| Level | Option | MobileNetV2 | ShuffleNetV2 | InceptionV3 |
|:---:|:---:|:---:|:---:|:---:|
| Backbone | Branches | 1 | 2 | 4 |
| Backbone | Blocks | 4 | 3 | 3 |
| Supernet | Input resolution | 32, 64, 96, 128 | 32, 64, 96, 128 | 32, 64, 96, 128 |
| Supernet | Width multiplier | 0.25, 0.5, 0.75, 1.0 | 0.2, 0.5, 0.8 | 0.25, 0.5, 0.75, 1.0 |
| Block | Kernel size | 3, 5 | 1, 3, 5, 7 | 3, 5, 7 |
| Block | Expansion / stride | expansion: 3, 4, 6 | stride: 1, 2 | stride: 1, 2 |
| Block | Layers | 1, 2, 3 | 1, 2, 3 | 1, 2, 3 |


### Accuracy
Below are the networks found by DupNAS, TinyTS, and PatchTS.

| Model | VM | DupNAS | TinyTS | PatchTS |  
|:---:|:---:|---:|:---:|:---:|
| MobileNetV2 | 96 KB | 58.40% | 52.88% | 51.36% | 
|             | 128 KB | 62.08% | 56.88% | 56.48% | 
|             | 256 KB | 62.64% | 61.76% | 62.64% | 
| ShuffleNetV2 | 96 KB | 61.36% | 56.8% | 54.24% | 
|             | 128 KB | 62.96% | 59.76% | 58.72% | 
|             | 256 KB | 65.76% | 64.96% | 60.48% | 
| InceptionV3 | 96 KB | 61.84% | 45.68% | 54.00% | 
|             | 128 KB | 64.16% | 57.84% | 58.64% | 
|             | 256 KB | 68.24% | 64.88% | 67.36% | 


<!-- | Model | VM | DupNAS | TinyTS | PatchTS |  Architecture | 
|---|---:|---:|---:|---:|---|
| MobileNetV2 | 96 KB | 58.40% | 52.88% | 51.36% | [Ori.](/assets/models/onnx_original/mbv2-vm96) / [TS](/assets/models/onnx_withTS/mbv2-vm96) |
|             | 128 KB | 62.08% | 56.88% | 56.48% | [Ori.](/assets/models/onnx_original/mbv2-vm128) / [TS](/assets/models/onnx_withTS/mbv2-vm128) |
|             | 256 KB | 62.64% | 61.76% | 62.64% | [Ori.](/assets/models/onnx_original/mbv2-v256) / [TS](/assets/models/onnx_withTS/mbv2-v256) |
| ShuffleNetV2 | 96 KB | 61.36% | 56.8% | 54.24% | [Ori.](/assets/models/onnx_original/shuffle-vm96) / [TS](/assets/models/onnx_withTS/shuffle-vm96) |
|             | 128 KB | 62.96% | 59.76% | 58.72% | [Ori.](/assets/models/onnx_original/shuffle-vm128) / [TS](/assets/models/onnx_withTS/shuffle-vm128) |
|             | 256 KB | 65.76% | 64.96% | 60.48% | [Ori.](/assets/models/onnx_original/shuffle-v256) / [TS](/assets/models/onnx_withTS/shuffle-v256) |
| InceptionV3 | 96 KB | 61.84% | 45.68% | 54.00% | [Ori.](/assets/models/onnx_original/incept-vm96) / [TS](/assets/models/onnx_withTS/incept-vm96) |
|             | 128 KB | 64.16% | 57.84% | 58.64% | [Ori.](/assets/models/onnx_original/incept-vm128) / [TS](/assets/models/onnx_withTS/incept-vm128) |
|             | 256 KB | 68.24% | 64.88% | 67.36% | [Ori.](/assets/models/onnx_original/incept-vm256) / [TS](/assets/models/onnx_withTS/incept-vm256) |
 -->




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

