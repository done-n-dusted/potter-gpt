# potter-gpt
A from-scratch-transformer model to generate a story based on the bunch of stories given as input. The model is highly customizable handling variety of hyper parameters. The preliminary code trains in a Apple Macbook Air M1 also. 

## Setup

### Requirements
The required packages can be found in `requirements.txt` file. The python version used is `Python 3.10.6`. Run the following commands to setup the repository to train and test the code.
```
$ python -m venv <env>
$ source <env>/bin/activate
$ pip install -r requirements.txt
```

### Configuration
The file `config.yml` contains the hyperparameters for training the transformer model. Each of the variables can be changed based on available resources.

### Dataset
* `config.yml` contains a variable named `data_path`. This variable correponds to the data used for training/testing the model.
* The path corresponds to the folder with a list of `.txt` files that contains text based on which you intend to train your transformer. You can find suitable text files from [Project Gutenberg](https://www.gutenberg.org/).
* The model was tested on Sir Arthur Conon Doyle's Sherlock Holmes Books:
    - Adventures of Sherlock Holmes
    - Memoirs of Sherlock Holmes
    - Return of Sherlock Holmes

## Training

### Tokenizer
`tokenizer.py` has a basic implementation for the tokenizer used. This is mainly a wrapper to the `AutoTokenizer` by HuggingFace. The model can be changed and configured in `config.yml` file. `gpt2` is the default model on for tokenizing the text with 50k+ vocab tokens.

### Transformer Model
`transformer.py` file contains the class `LanguageModel` that has the implementation of the whole decoder block of the Transformer. The model has a ton of sub components. In depth details can be found in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762). The code tries to replicate the Decoder block of the Transformer architecture. 

- **Head** module is the basic Attention head for the model. It finds the Q, K, V matrices required for attention.
- **MultiHead** module has multiple **Head** modules. The number of heads can be determined in the `config.yml` file. The class automatically splits the size for each of the heads based on the input size.
- **FeedForward** module is the Feed Forward network following MultiHead attention in a block. 
- **LanguageModel** Module contains the whole architecture. It is basically a set of blocks following each other. This class also has a `generate_sequence` method that returns the embedding of the following words which can be decoded with the above defined `tokenizer`.

## Main
`main.py` contains the overall wrapper of the entire architecture. The training loop takes in hyperparameters from the `config.yml` file and trains the model. It prints the loss and an inference *50* times in total regardless of the number of steps taken. Run the whole transformer after the setup with the following command:
```
$ python main.py
```

## Logging
Logging can be controlled in the `config.yml` file. `save_model` saves the model in `results/` folder. `log_name` corresponds to the `csv` file in which the log is saved.

## References
1. Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
2. HuggingFace Transformer's [AutoTokenizer](https://huggingface.co/docs/transformers/v4.15.0/en/model_doc/auto#transformers.AutoTokenizer).
3. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
4. Books to Harry Potter - [Kaggle](https://www.kaggle.com/datasets/moxxis/harry-potter-lstm).
5. [Project Gutenberg](https://www.gutenberg.org/)