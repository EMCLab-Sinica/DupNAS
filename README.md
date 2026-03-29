# 🚀 DupNAS: Splitting Bottlenecks: Memory-Aware Neural Architecture Search for Multi-Branch TinyML

> **Official artifact repository for DupNAS**  
> Memory-constrained neural architecture search, tensor splitting, and deployment for TinyML on MCU-class devices.


<!-- [![Paper](#)](#) [![License](#)](#) [![Python](#)](#) [![Platform](#)](#) -->

---
## 📝 Overview

DupNAS is a framework for improving neural network accuracy under tight memory constraints on resource-constrained devices. It combines neural architecture search with multi-branch tensor splitting to reduce peak memory usage and make larger or more accurate networks deployable on small devices.

To address the peak memory challenge, DupNAS greatly shrinks the configuration space and incrementally explores only the configurations that can remove memory bottlenecks with low latency overhead.

We build DupNAS by integrating our multi-branch splitting method into TinyNAS for microcontrollers. We evaluate it on several vision-based TinyML network families under different memory budgets, and also deploy the searched networks on an STM32 microcontroller.

DupNAS is implemented in PyTorch and developed on a server with an Intel Xeon E5-2678 CPU (2.5GHz), 128 GB RAM, and four NVIDIA GTX 1080Ti GPUs. The split-network solutions are INT8-quantized and deployed on an STM32F746 MCU with an ARM Cortex-M7 CPU 216 MHz), 320 KB VM, and 1 MB NVM, running the TFLite Micro inference engine. We also extend the TFLite Micro model converter to avoid extra NVM usage caused by duplicated weights when converting split networks from PyTorch to the deployment format.

We evaluate DupNAS on three backbone network families—MobileNetV2, ShuffleNetV2, and InceptionV3—trained on the ImageNet-100 dataset. We compare DupNAS with two existing splitting methods, TinyTS and PatchTS.

<p align="center">
  <img src="assets/figures/NAS_with_TS.svg" alt="NAS_with_TS" width="800">
</p>
<p align="center">
  <em>Overview of the DupNAS</em>
</p>
<!-- This repository contains the full artifact for reproducing the NAS, model splitting, fine-tuning, ONNX export, and MCU deployment workflow used in DupNAS. -->


---
## 📌 Directory/File Structure

Below is a brief description of the main directories and files in this repository.

- `/DupNAS/NASBase/duplication` implements the DupNAS module integrated into the NAS framework.
- `/DupNAS/NASBase/ss_optimization` contains the search-space optimization component, adapted from TinyNAS .
- `/DupNAS/NASBase/evo_search` contains the evolutionary search component, adapted from TinyNAS.
- `/DupNAS/NASBase/model` defines the search space, supernet architecture, and subnet architecture.
- `/DupNAS/settings` provides the settings used for evaluation under different datasets and baseline methods.
- `/DupNAS/settings.py` defines the global NAS settings and provides utilities for loading and managing configuration files.
- `/DupNAS/genonnx/DupNAS_SA.py` provides a standalone implementation of the DupNAS module.
- `/Inference/Tensor-splitter/` implements ONNX model rewriting to execute the selected tensor-splitting configuration.
- `/Inference/Tflm-engine/` implements the build process for TensorFlow Lite Micro libraries that run the models.
- `/assets/DupNAS_paper_data.xlsx` contains the data presented in the figures in the paper.

---
## 🧭 Getting Started


### 💡 Requirement

- `Python 3.9` is recommended.
- Install the required Python packages listed in `requirements.txt` with:
  `python3.9 -m pip install -r requirements.txt`
- [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install/overview) is optional, but recommended for managing Python environments.
- The main dataset used in this project is [ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100/data). You can prepare and load it using: `/DupNAS/NASBase/load_image100.py`.
- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)
- [STM32F746NG MCU](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)

### 🔧Setup and Build for DupNAS

1. Download or clone this repository
2. Create and activate a Python environment, then install the required dependencies.
3. Prepare the ImageNet-100 dataset and update the dataset paths.
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
  | `--mode` | TS optimization  | `dupnas`, `tinyts`, `patchts`, `nots` | 
  | `--vmsize` | VM constraint in KB | `96`, `128`, `256` |
  | `--suffix` | Experiment suffix for output naming | user-defined string |
  
5. Extract the solution from `best_solution,json` and use it to fill in `spec_model.txt`
6. Generate the ONNX models for the selected solution using `/DupNAS/NASBase/spec_onnx_gen.py`. The outputs will be saved in `/DupNAS/genonnx/`.
7. Go to `/DupNAS/genonnx/`, then run `python3.9 -m DupNAS_SA.py --onnx <onnx_name> --mode <mode> --vmsize <vmsize> --export_file` to generate the TS configuration. Alternatively, you can run `run_all_onnx.sh` to automatically generate TS configurations for all ONNX models in the directory.
8. Run `gen_ts_cfg.py` to collect the `split-configuration JSON file` for Tensor Splitter


### ✂️ Tensor Splitter
1. Copy the ONNX model and its corresponding split-configuration JSON file from `/DupNAS/genonnx/` to `/Inference/Tensor-splitter/`
2. For more information, please refer to [Tensor-splitter/README.md](Tensor-splitter/README.md).


### ⚙️ Inference

To deploy models with [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro) on STM32, follow the steps below:

1. Convert the ONNX models to TFLite with [onnx2tf](https://github.com/PINTO0309/onnx2tf). One convenient option is to use the official Docker image:
   ```bash
   run --rm -it -v $(pwd):/workdir -w /workdir ghcr.io/pinto0309/onnx2tf:1.28.5  
   onnx2tf -i ONNX_MODEL -oiqt
   ```
   This produces fully integer-quantized TFLite models such as `xxx_full_integer_quant.tflite`.

2. Copy the converted TFLite model (`xxx_full_integer_quant.tflite`) into `Tflm-engine/src/models`.

3. Follow [Tflm-engine/README.md](Tflm-engine/README.md) to build the TensorFlow Lite Micro static library (`libtensorflow-microlite.a`).

4. Add the generated static library to your STM32CubeIDE project settings. Then include `Tflm-engine/src/tflm_main.h` and call `tflm_main_xxx` to run inference for the target model.

For more information, please refer to [Tflm-engine/README.md](Tflm-engine/README.md).


<!-- sudo docker run --rm -it -v $(pwd):/workdir -w /workdir ghcr.io/pinto0309/onnx2tf:1.28.5   -->

---
## 🧩 Results
For more detailed data, please see [DupNAS_paper_data](/assets/DupNAS_paper_data.xlsx)

### Accuracy

| Model | TS Mode | VM = 96 KB | VM = 128 KB | VM = 256 KB |
|---|---|---:|---:|---:|
| MobileNet   | DupNAS  | 58.40% | 62.08% | 62.64% |
|             | TinyTS  | 52.88% | 56.88% | 61.76% |
|             | PatchTS | 51.36% | 56.48% | 62.64% |
| ShuffleNet  | DupNAS  | 61.36% | 62.96% | 65.76% |
|             | TinyTS  | 56.8% | 59.76% | 64.96% |
|             | PatchTS | 54.24% | 58.72% | 60.48% |
| InceptionNet | DupNAS  | 61.84% | 64.16% | 68.24% |
|             | TinyTS  | 45.68% | 57.84% | 64.88% |
|             | PatchTS | 54.00% | 58.64% | 67.36% |



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

