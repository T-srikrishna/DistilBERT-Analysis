# AIDI 1002 Final Project: An Empirical Analysis of DistilBERT

This repository contains the final project for the AIDI 1002 course. This project reproduces the findings of the paper **"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"** and extends the analysis with two significant contributions.

## Project Objectives

The project is structured as a three-part experiment:

1.  **Reproducibility:** To reproduce the original paper's text classification performance by fine-tuning DistilBERT on the IMDb sentiment dataset.
2.  **Contribution 1 (Generalization):** To evaluate the model's effectiveness in a new context by testing it on the SST-2 dataset, which contains shorter text samples.
3.  **Contribution 2 (Efficiency Analysis):** To compare the performance and computational cost of DistilBERT against a fast, classical machine learning baseline (Logistic Regression with TF-IDF features).

## Repository Structure

-   `DistilBERT-Analysis.ipynb`: The main Jupyter Notebook containing all code, analysis, and results.
-   `requirements.txt`: A list of all necessary Python libraries to run the project.
-   `README.md`: This file, providing an overview and instructions.

## How to Run the Experiments

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/T-srikrishna/DistilBERT-Analysis.git
    cd DistilBERT-Analysis
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: A GPU is highly recommended for the deep learning experiments. Ensure you have a version of PyTorch compatible with your CUDA drivers if you plan to use a local GPU.*

4.  **Run the Jupyter Notebook:**
    Launch Jupyter Notebook and open the `DistilBERT-Analysis.ipynb` file.
    ```bash
    jupyter notebook
    ```
    Execute the cells from top to bottom. The notebook is designed to run all three experiments sequentially.

## Summary of Findings

Our results provide a comprehensive, data-driven view of DistilBERT's capabilities:

-   **Reproduction (IMDb):** We successfully reproduced the paper's findings, achieving **91.84%** accuracy in **56.80 minutes**.
-   **Generalization (SST-2):** The model proved to be robust, achieving **91.06%** accuracy on the SST-2 dataset in **38.38 minutes**, demonstrating its ability to handle different text formats.
-   **Efficiency (Classical Baseline):** A classical Logistic Regression model achieved a highly competitive accuracy of **89.52%** on IMDb in just **0.56 seconds**, highlighting the dramatic trade-off between peak performance and computational cost.

*Detailed analysis, code, and output for all experiments are documented inside the Jupyter Notebook.*

## Acknowledging the Original Work

This project is a direct implementation and extension of the research presented in the DistilBERT paper. All credit for the original methodology and model architecture belongs to the authors. We highly recommend reading their original work for a deeper understanding.

-   **Paper:** [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
-   **Authors:** Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf..
