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

To run the training script, you need to provide two arguments: `-epochs` which specifies the number of epochs for training, and `-path_to_file` which specifies the path to your dataset file. Here is how you can run the script from the command line:

```bash
python train.py -epochs <number_of_epochs> -path_to_file <path_to_your_dataset>
```
