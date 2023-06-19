# Tumor Droplet Classification <br> <sub><sup> | Data Science Lab | ETH Zurich | Fall 2023 | </sup></sub>

This repository contains the code for an automated tumoroid viability classification pipeline, designed for personalized cancer treatment. The pipeline consists of a comprehensive data preprocessing pipeline that handles microscopic images of droplet collections containing tumoroids. It performs droplet extraction, image normalization, and foreground extraction. The classification task is accomplished using a pretrained Convolutional Neural Network (CNN) that has been fine-tuned on a dataset specifically curated for cancer tumoroids. To ensure reliable training, the dataset has been carefully filtered to remove outliers.

This repository also includes a statistical analysis demonstrating how medication choices can be informed by the model in a reliable manner. Our work demonstates how medical decision can be significantly improved by taking predictions over multiple tumoroids from a single patient; through statistical analysis, we can provide strong statistical guarantees for the effectiveness of a drug, only using minimal assumptions.

**Key Features:**
* Automated tumoroid viability classification pipeline
* Data preprocessing: droplet extraction, image normalization, and foreground extraction
* Pretrained CNN for tumoroid classification
* Fine-tuning on a dataset of colon cancer tumoroids
* Statistical analysis for medication choice based on the model

We hope that this repository will contribute to advancing personalized cancer treatment by providing a reliable and automated method for tumoroid viability classification. Feel free to explore the code and adapt it to your specific needs.

**Note:** This repository focuses on the code implementation. For a detailed description of the research findings and methodology, please refer to the accompanying [report](DataScienceLab-report.pdf) and [slides](DSLab_demo.pdf).


## Acknowledgements

This is the repository was created as part of the Data Science Lab at ETH Zürich in Fall 2022 by Johan Lokna, Fredrik Nestaas, Viggo Moro, Florian Hübler, Tobias Wegel.
We would also like to thank our supervisor Ines Luechtefeld


## Setup

Necessary software: The code has been tested with Python 3.8 and above.

1. Create a virtual environment with python 3.8 or above.
2. Clone the repository.
3. install the required libraries using ```pip install -r requirements.txt```.

You should be good to go!

## Usage

Usage is meant to be done mainly through ```main.ipynb```. 
