# Emergent-Language-Symbolic-Autoencoder
Official Implementation of Emergent Language Symbolic Autoencoder (ELSA)


## Setting Up the Environment

To run this project, you need to recreate the Conda environment. First, make sure you have Conda installed. Then, navigate to the root directory of this project in your terminal and run:

```bash
conda env create -f environment.yml
```

## Dataset
To access the dataset, use [this link](https://drive.google.com/file/d/10m99jqJe9gJxu8rfTte3DaiFdu6XbmcQ/view?usp=sharing).

## Running the Training Script

To run the training script, you should provide these arguments:

- `-epochs <int>`: Number of epochs for training (required).
- `-path_to_file <str>`: Path to your dataset file (required).
- `-loss_mode <int>`: Choose the loss function mode (default = `4`):
  - `1` = Regular
  - `2` = Progressive
  - `3` = Progressive Strict
  - `4` = Progressive Strict with Containing Bias
- `-sentence_length <int>`: Set the sentence length (default = `5`).
- `-vocab_size <int>`: Set the vocabulary size (default = `100`).

### Example usage:
```bash
python train.py -epochs 700 -path_to_file ./mydataset.pkl -loss_mode 4 -sentence_length 5 -vocab_size 100
```
