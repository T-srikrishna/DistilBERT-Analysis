# AIDI 1002 Final Project: An Empirical Analysis of DistilBERT

This repository contains the final project for the AIDI 1002 course. This project reproduces the findings of the paper **"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"** and extends the analysis with two significant contributions.

## Project Objectives

The project is structured as a three-part experiment:

1. **Reproducibility:** To reproduce the original paper's text classification performance by fine-tuning DistilBERT on the IMDb sentiment dataset.  
2. **Contribution 1 (Generalization):** To evaluate the model's effectiveness in a new context by testing it on the SST-2 dataset, which contains shorter text samples.  
3. **Contribution 2 (Efficiency Analysis):** To compare the performance and computational cost of DistilBERT against a fast, classical machine learning baseline (Logistic Regression with TF-IDF features).

## Repository Structure

- `distilbert_vs_bert_experiment.ipynb`: The main Jupyter Notebook containing all code, analysis, and results. This serves as the **Project Report**.
- `requirements.txt`: A list of all necessary Python libraries to run the project.
- `README.md`: This file, providing an overview and instructions.

## How to Run the Experiments

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/T-srikrishna/bert-vs-distilbert.git
    cd bert-vs-distilbert
    ```

2. **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    > **Note:** A GPU is highly recommended for the deep learning experiments.  
    > If you're using an NVIDIA GPU, install the CUDA-enabled PyTorch wheel:
    > ```bash
    > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    > ```

4. **Run the Jupyter Notebook:**
    Launch Jupyter Notebook and open the `distilbert_vs_bert_experiment.ipynb` file.
    ```bash
    jupyter notebook
    ```
    Execute the cells from top to bottom. The notebook is designed to run all three experiments sequentially.

## GPU Acceleration (Tested on Local Hardware)

This project was tested on a system with an **NVIDIA GeForce RTX 4060 Laptop GPU**.  
On this setup, fine-tuning DistilBERT for 2 epochs on the IMDb dataset with a batch size of 16 took approximately **X minutes**. (Update this based on your run time.)

If running on CPU, training is still possible but significantly slower.

## Python Version

This project was developed and tested using **Python 3.11**.

## Summary of Findings

Our results provide a comprehensive view of DistilBERT's capabilities:

- **Reproduction:** We successfully reproduced the paper's findings, achieving **91.84%** accuracy on the IMDb dataset, confirming the model's effectiveness.  
- **Generalization:** The model performed well on the SST-2 dataset, demonstrating its ability to generalize to different text formats.  
- **Efficiency:** A Logistic Regression baseline with TF-IDF achieved strong performance in under a minute, emphasizing the trade-off between efficiency and deep learning performance.

*Full results and analysis are documented inside the Jupyter Notebook.*

## Acknowledging the Original Work

This project builds on the research from the DistilBERT paper, which introduced a lighter and faster version of BERT while keeping most of its performance.  
Since the goal was to reproduce and extend their findings, it’s only fair to give credit.

If you're using or referencing this project in your own work, please consider citing the original authors:

```bibtex
@inproceedings{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  booktitle={NeurIPS EMCML},
  year={2019}
}
