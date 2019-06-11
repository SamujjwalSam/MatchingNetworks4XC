# End-to-End Deep Learning Framework for Extreme Classification using Matching Networks (MNXC)[![Build Status](https://travis-ci.org/vishwakftw/extreme-classification.svg?branch=master)](https://travis-ci.org/vishwakftw/extreme-classification)

`MNXC` is a Python module designed for Extreme Multi-Label Classification tasks using Matching Networks.
This project also includes scripts for training and testing on datasets (Wiki-31k, Amazon-13k, Amazon-14k) using this module.

## Setup

### Prerequisites

- Python 3.6 (Not tested in Python 2.7)
- Requirements for the project for `linux-64` and `Windows` are listed in [linux64-requirements.txt](linux64-requirements.txt) and [win-requirements.txt](win-requirements.txt) respectively. In addition, [linux64-spec-file.txt](linux64-spec-file.txt) and [win-spec-file.txt](win-spec-file.txt) contains the requirements in `explicit` form.
- PyTorch 1.0 or higher is necessary. The requirements can be installed using pip:
   ```bash
   $ pip install -r requirements.txt 
   ```
   or using conda:
   ```bash
   $ conda install --file requirements.txt
     ```
     
<!---
### Installing `MNXC`

- Clone the repository.
  ```bash
  $ git clone https://github.com/vishwakftw/extreme-classification
  $ cd extreme-classification
  ```

- Install
  ```bash
  $ python setup.py install
  ```

- To test if your installation is successful, try running the command:
  ```bash
  $ python -c "import MNXC"
  ```
--->

## Details about the scripts

### MNXC_main

`MNXC_main.py` runs the Matching Networks with `layer_size = num_channels`. A description of the options available can be found using:

```bash
python MNXC_main.py --help
```

This script trains (and optionally evaluates) evaluates a model on a given dataset using Matching Networks.

### MNXC_main_orig

Use `MNXC_main_orig.py` runs the Matching Networks with `layer_size = num_channels = 1`. A description of the options available can be found using:
```bash
python MNXC_main_orig.py --help
```
This script trains (and optionally evaluates) evaluates a model on a given dataset using the Matching Networks algorithm.

## Variable Naming Convention:
Below is the naming convention followed in the project. All files saved will be prepended with dataset name.

    Name used           ->  Meaning
    --------------------------------------------------------
    Categories          ->  Labels / Classes {}
    Sample              ->  [Feature, Categories]
    *_hot               ->  any multi-hot vector label
    x_hat               ->  any test feature batch or sample
    no_cat_ids [list]   -> doc_id's for which no categories were found.

## Data Format
The default input data should be in json format. This project uses the "raw data" available [here](http://manikvarma.org/downloads/XC/XMLRepository.html) which can be converted using the scripts available in `data_loaders` directory. The scripts can handle 3 types of data i.e. html, json and txt format. Internal structures of the generated json files are as follows:

    _train: Training part of data.
    
    _test: Testing part of data.
    
    _val: Validation part of data.

    sentences : Doc_id to texts mapping after parsing and cleaning.
    sentences =  {
                    "doc_id1": "text_1",
                    "doc_id2": "text_2"
                 }
    
    classes   : OrderedDict of doc_id to classes mapping.
    classes =    {     
                    "doc_id1" : [cat_id_1, cat_id_2],
                    "doc_id2" : [cat_id_2, cat_id_10]
                 }
    
    categories : OrderedDict of cat_text to cat_id mapping.
    categories = {
                    "cat_text"          : cat_id
                    "Computer Science"  : cat_id_1,
                    "Machine Learning"  : cat_id_2
                 }
                 
    cat_id2text_map: Opposite of "categories", category id (str) to text map.
    cat_id2text_map = {
                    "cat_id_1"  : "cat_text"
                    "0"         : "Computer Science",
                    "1"         : "Machine Learning"
                 }
    
    dup_cat_text_map: OrderedDict of duplicate cat_text to original cat_text. Different categories which become same after cleaning.
    dup_cat_text_map = {
                    "Cat_Text"              : "cat_text",
                    "Business Intelligence" : "business intelligence",
                    }
                    
    categories_dup_dict: OrderedDict of duplicate cat_id to original cat_id. Different categories which become same after cleaning.
    categories_dup_dict = {
                    "cat_id"    : cat_id,
                    "10248"     : 3405,
                    }
    
    idf_dict: Mapping of tokens to their idf score.
    idf_dict = {    
            "token" : idf_score
            "!"     : 1.3208941423848382,
            "#"     : 4.731611284387589,
            }
    
    hid_categories: Same as categories, but not used in the project. Present only in Wiki datasets.
    hid_categories = {
            "cat_text"                                  : cat_id,
            "All articles with unsourced statements"    : 0
            }
    
    hid_classes:
    hid_classes = {
            "doc_id1"                           : [cat_id_1, cat_id_2],
            "4a50553e1be4f81bba91849caa9a59c0"  : [ 0, 1, 2, 3, 4],
            }                 

## System and Model Configurations
We used a Python file called `config.py` to store the configuration and other relevant information. It has 3 global variables in the form of Python dict and two functions to retrieve dynamic system information such as os version, username, etc.

### configuration
For using NeuralXC through `train_neuralxc.py`, you need to have valid neural network configurations for the autoencoders of the inputs, labels and the regressor in the YAML format. An example configuration file is:
```json
{
"data":         "here",
"xc_datasets":  "here",
"model":        "here",
"lstm_params":  "here",
"cnn_params":   "here",
"sampling":     "here",
"prep_vecs":    "here",
"text_process": "here",
"paths":        "here",
"html_parser":  "here"
}

```
Please note that the `name` and `kwargs` attributes have to resemble the same names as those in PyTorch.

### seed
Optimizer configurations are very similar to the neural network configurations. Here you have to retain the same naming as PyTorch for optimizer names and their parameters - for example: `lr` for learning rate. Below is a sample:
```yaml
name: Adam
args:
  lr: 0.001
  betas: [0.5, 0.9]
  weight_decay: 0.0001
```

### Dataset Configurations
In both the scripts, you are required to specify a data root (`data_root`), dataset information file (`dataset_info`). `data_root` corresponds to the folder containing the datasets. `dataset_info` requires a YAML file in the following format:
```yaml
train_filename:
train_opts:
  num_data_points:
  input_dims:
  output_dims:

test_filename:
test_opts:
  num_data_points:
  input_dims:
  output_dims:
```

If the test dataset doesn't exist, then please remove the fields `test_filename` and `test_opts`. An example for the Bibtex dataset would be:
```yaml
train_filename: bibtex_train.txt
train_opts:
  num_data_points: 4880
  input_dims: 1836
  output_dims: 159

test_filename: bibtex_test.txt
test_opts:
  num_data_points: 2515
  input_dims: 1836
  output_dims: 159
```

## TODOs (Ordered by priority): 
1. Implement TF-IDF weighted vectors
2. Prepare Delicious-T140 dataset

## Tricks to solve few reoccurring problems:
#### To solve MKL problem:
Adding <conda-env-root>/Library/bin to the path in the run configuration solves the issue, but
adding it to the interpreter paths in the project settings doesn't.
https://stackoverflow.com/questions/35478526/pyinstaller-numpy-intel-mkl-fatal-error-cannot-load-mkl-intel-thread-dll
 

## License
This code is provided under the [MIT License](LICENSE)

---
This project is work in progress.
