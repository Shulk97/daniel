# DANIEL: A Fast Document Attention Network for Information Extraction and Labeling of Handwritten Documents

This repository contains the official implementation of the paper:

**"DANIEL: A Fast Document Attention Network for Information Extraction and Labeling of Handwritten Documents"**
from Thomas CONSTUM, Pierrick TRANOUEZ and Thierry PAQUET (LITIS, University of Rouen Normandie).

The paper has been accepted for publication in the [International Journal on Document Analysis and Recognition (IJDAR)](https://link.springer.com/article/10.1007/s10032-024-00511-9) and is also available on [arXiv](https://arxiv.org/abs/2407.09103).

This repository includes:
- The training and inference code.
- A synthetic data generator.

A demonstration video of DANIEL is available on [YouTube](https://youtu.be/ibJJrkYMl1U).

Pretrained model weights can be downloaded [here](https://zenodo.org/records/15633064).

**This project is licensed under a custom Research Usage Only (RUO) license. Please refer to the license file LICENSE for more details.**

## Table of Contents
1. [Getting Started](#getting-started)
2. [Reproducing Results](#reproducing-results)
3. [Training DANIEL on Your Own Dataset](#training-daniel-on-your-own-dataset)
4. [Choosing Transfer Learning Weights](#choosing-transfer-learning-weights)
5. [Project Structure](#project-structure)
6. [Training Parameters](#training-parameters)

## Getting Started

### Environment Setup

- **CUDA**: Version 12 is strongly recommended, along with an NVIDIA GPU with at least **16GB VRAM** for inference and **80GB VRAM** for training.
- **Python**: The recommended version is **3.9**. If a different version is used, a Conda environment should be created to ensure compatibility.
- **Conda**: Strongly recommended for environment replication. [Installation instructions](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)

#### Installation Steps:
```bash
conda create --name daniel-env python=3.9
conda activate daniel-env
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip3 install -r requirements.txt
git clone https://gitlab.teklia.com/ner/nerval.git
cd nerval
git checkout 0.3.1
pip3 install .
```

### Required Files

Certain files are necessary for running DANIEL and can be downloaded from [Zenodo](https://zenodo.org/records/15607305):
- **Tokenizer**: The folder `tokenizer-daniel` should be placed in `basic/subwords`.
- **Substitution Dictionary**: `replace_dict.pkl`, which contains substitution candidates for each subword during teacher forcing. Place this file in `basic/subwords`.

## Reproducing Results

DANIEL has been evaluated for:
- **HTR (Handwritten Text Recognition)** on **READ 2016, RIMES 2009, IAM, and M-POPP**.
- **NER (Named Entity Recognition)** on **IAM NER and M-POPP NER**.

### Step 1: Obtain the Datasets

Formatted datasets should be placed in `Datasets/formatted`. The label format follows the DAN format.

For **READ 2016** and **RIMES 2009**, follow the dataset preparation guide in the [DAN repository](https://github.com/FactoDeepLearning/DAN).

For **M-POPP/M-POPP NER**, formatted labels are available on [Zenodo](https://zenodo.org/records/12737132).

For **IAM**, use the formatting script:
```bash
python3 Datasets/dataset_formatters/iam_formatter.py
```
IAM dataset files can be found [here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database).

### Step 2: Download Model Weights

Pretrained weights can be downloaded from the provided link. Extract them into the `outputs` folder:
```bash
outputs/daniel_datasetname_strategy_X
```

### Step 3: Model Evaluation

Run the appropriate script based on the dataset and strategy:
```bash
python3 OCR/document_OCR/daniel/<dataset>/<task>/daniel_<dataset>_strategy_<X>.py
```
For example, to evaluate a model on **M-POPP NER** with strategy C:
```bash
python3 OCR/document_OCR/daniel/mpopp/ner/daniel_mpopp_ner_strategy_C.py
```

## Training DANIEL on Your Own Dataset

To adapt DANIEL to a new dataset, use the fine-tuning script:
```bash
python3 OCR/document_OCR/daniel/custom_dataset/daniel_custom_dataset_fine_tuning.py
```
This script performs transfer learning from a DANIEL model trained on M-POPP with strategy A.
You should therefore download the corresponding weights.

### Dataset Requirements
- Place your dataset in `Datasets/formatted/custom_dataset_page_sem` with images in `train`, `valid`, and `test` folders.
- Formatted labels must be named `labels-custom-dataset.pkl`. For more information on the required format, please refer to the existing formatter scripts or the formatted labels of M-POPP available on Zenodo.
- If using **semantic tokens**, update:
  - `basic/post_processing_layout.py` with a new `PostProcessingModuleDatasetName` class following the same format as the PostProcessing classes of the other datasets. This class add a post processing step used when computing metrics based on semantic tokens. For instance, it forces an opening tag from a class to be followed by a closing tag from the same class before opening another tag.
  - `basic/metric_manager.py` with a new `str_to_graph_dataset_name` function, following the same format as the `str_to_graph` functions of the other datasets. This function is used when computing the MAP_CER metric.
- If using **named entities**:
  - you must define a format to encode them in the text. DANIEL supports the following formats:
    - **after**: Semantic tokens are placed **after** the corresponding words. If a named entity spans multiple words, the entity tag should be placed after each word.
    - **before**: Semantic tokens are placed **before** the corresponding words. If a named entity spans multiple words, the entity tag should be placed before each word.
  - You need to add or modify several parameters in the `params` dictionary in the training script:
    - Add the parameter `"tagging_mode"` in `"dataset_params"` with the value `"before"` or `"after"`, depending on your chosen format.
    - In `"training_params"`:
      - Set the parameter `"focus_metric"` to `"nerval"`, as this metric is used to evaluate NER performance.
      - Set the parameter `"expected_metric_value"` to `"high"`, meaning that higher metric values during evaluation will be considered better, and the corresponding training weights will be saved.
      - Add the metric `"cer-ner"` to `"train_metrics"`. This metric computes the edit distance between the named entity tokens in the ground truth and the prediction, ignoring regular characters. It is used during training because it is faster to compute than `"nerval"`.
      - Add the metric `"nerval"` to `"eval_metrics"`.

### Advanced Strategy: Using Custom Synthetic Data

To leverage synthetic data, you need to gather the following elements:

- **Fonts** used for rendering text in synthetic images:
  - You can obtain these using the following scraper: [synthetic_text_gen](https://github.com/herobd/synthetic_text_gen/tree/master).
- **Corpus** used for generating synthetic text:
  - the processed Wikipedia corpus are available on [Zenodo](https://zenodo.org/records/15629573).

Once all necessary elements are collected, follow these steps to integrate synthetic data into your training pipeline:

1. **Adapt the code to use the script**:
   `OCR/document_OCR/daniel/custom_dataset/daniel_custom_dataset_custom_synth_fitting.py`
   - This script applies transfer learning, starting from DANIEL trained on M-POPP (Strategy A) but fine-tuned exclusively on your custom synthetic data.
   - To make this script functional, you must generate your own synthetic dataset. You can find example implementations at:
     - [`OCR/synth_doc/synth_doc_generation.py#L1179`](OCR/document_OCR/daniel/synth_doc/synth_doc_generation.py?ref_type=heads#L1179) – IAM dataset (simple synthetic data).
     - [`OCR/synth_doc/synth_doc_generation.py#L824`](OCR/document_OCR/daniel/synth_doc/synth_doc_generation.py?ref_type=heads#L824) – M-POPP dataset (more complex data).
   - Once your dataset is ready, add an entry for it in:
     [`OCR/ocr_dataset_manager.py#L664`](OCR/ocr_dataset_manager.py?ref_type=heads#L664).
   - Start training and monitor the Character Error Rate (CER). Aim for a training CER around 5% or lower. Experiment with different training durations to determine the optimal point for transitioning to the next step.

2. **Fine-tune using the script**:
   `OCR/document_OCR/daniel/custom_dataset/daniel_custom_dataset_fine_tuning_with_synth_data.py`
   - This script performs transfer learning using the model weights from the previous training step.
   - Locate the following section in the script and replace `best_X.pt` with the actual filename of the best weights produced in the previous step:
     ```python
     "transfer_learning": {
       # model_name: [state_dict_name, checkpoint_path, learnable, strict]
       "encoder": ["encoder", "outputs/daniel_custom_dataset_custom_synth_fitting/checkpoints/best_X.pt", True, True],
       "decoder": ["decoder", "outputs/daniel_custom_dataset_custom_synth_fitting/checkpoints/best_X.pt", True, False],
     },
     ```
   - Once the modifications are made, simply launch the training script.


## Choosing Transfer Learning Weights

### Selecting Pre-Trained Weights for Transfer Learning

When performing transfer learning, choosing the right pre-trained weights is crucial for achieving optimal results. Below are the recommended weight options based on your dataset and annotation availability:

#### 1. **`daniel_iam_ner_strategy_A_custom_split`**
   - **Training Data:** Trained on all synthetic datasets and real datasets *except* M-POPP.
   - **Best Use Case:** Suitable when only a small amount of annotated data is available in the target dataset.
   - **Attention Granularity:** 32-pixel vertical granularity, meaning the encoder’s output feature map has a height of **H/32** (where **H** is the input image height).

#### 2. **`daniel_mpopp_ner_strategy_A`**
   - **Training Data:** Trained on all synthetic datasets and real datasets, *including* M-POPP.
   - **Best Use Case:** Recommended when limited annotated data is available in the target dataset, especially for large images with small text.
   - **Attention Granularity:** 16-pixel vertical granularity (**H/16**), providing finer attention than the previous model.
   - **Performance Consideration:** Due to the finer granularity, this model is slower than `daniel_iam_ner_strategy_A_custom_split` but better suited for handling detailed text in large images.

#### 3. **`daniel_multi_synth`**
   - **Training Data:** Trained exclusively on synthetic datasets *excluding* M-POPP, with no real data. Used to initialize fine-tuning strategies **A and B** for the **IAM/IAM NER, RIMES 2009, and READ 2016** datasets.
   - **Best Use Case:** Suitable for modern document datasets with several thousand annotated pages.
   - **Attention Granularity:** 32-pixel vertical granularity (**H/32**).

#### 4. **`daniel_multi_synth_mpopp`**
   - **Training Data:** Trained exclusively on synthetic datasets *including* M-POPP, with no real data. Used to initialize fine-tuning strategies **A and B** for **M-POPP/M-POPP NER** datasets.
   - **Best Use Case:** Suitable for modern document datasets with several thousand annotated pages with a small text size compared to the size of the image.
   - **Attention Granularity:** 16-pixel vertical granularity (**H/16**).
   - **Performance Consideration:** Like `daniel_mpopp_ner_strategy_A`, this model has a finer attention granularity, making it slower but more effective for large images with small text.

## Generating Synthetic Data Offline

The full-document synthetic data used for training the DANIEL model is generated on-the-fly during training.
However, if you'd like to generate this data offline, for instance to visualize the data for debugging the generation, you can easily do so by launching any training script with the argument `mode='synth'`.

In the `specific_dataset_cfg` dictionnary, you can customize the synthetic data generation using the following parameters:

- **nb_samples**:
  Number of synthetic samples to generate.
- **nb_steps**:
  Simulate the number of training steps already performed when generating synthetic data offline.
- **synth_output_folder**:
  Path to the folder where the generated synthetic documents will be saved.


## Project Structure

The project is organized into the following directories:

- **`basic/`** – Contains the project's core files and utility scripts.
- **`Datasets/`** – Stores datasets along with scripts for converting raw datasets into the DAN format.
- **`Fonts/`** – Contains font files used for synthetic data generation.
- **`OCR/`** – Includes training scripts and model architecture definitions.
- **`outputs/`** – Stores trained model weights, TensorBoard logs, and prediction files. Each script specifies an `output_folder` parameter that determines the output location:
  - **`outputs/*output_folder*/results/`** – Contains TensorBoard logs, evaluation metrics, and predictions from evaluation runs.
  - **`outputs/*output_folder*/checkpoints/`** – Stores the trained model weights:
    - **`best_X.pt`** – Weights from the epoch that achieved the best validation CER.
    - **`last_X.pt`** – Weights from the last completed training epoch.


## Training parameters

All training parameters are explained in the comments of the files located in the `conf` and `OCR/document_OCR/daniel/custom_dataset` folders.

## Citation
```bibtex
@article{Constum2025,
  author = {Constum, Thomas and Tranouez, Pierrick and Paquet, Thierry},
  year = {2025},
  month = {01},
  pages = {1-23},
  title = {DANIEL: A Fast Document Attention Network for Information Extraction and Labeling of Handwritten Documents},
  journal = {International Journal on Document Analysis and Recognition (IJDAR)},
  doi = {10.1007/s10032-024-00511-9}
}
```