# GeNN
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/FahedSabellioglu/genn/blob/master/LICENSE.txt)

GeNN (generative neural networks) is a high-level interface for text applications using PyTorch RNN's.


## Features

1.  Preprocessing: 
	- Parsing txt, json, and csv files.
	- NLTK, regex and spacy tokenization support.
	- GloVe and fastText pretrained embeddings, with the ability to fine-tune for your data.
2. Architectures and customization:
	- LSTM and GRU, with variable size.
	- Variable number of layers and batches.
	- Dropout.
3. Text generation:
	- Random seed sampling from the n first tokens in all instances, or the most frequent token.
	- Top-K sampling for next token prediction with variable K.
	- Nucleus sampling for next token prediction with variable probability threshold.

## Getting started

### How to install
```bash
pip install genn
```
### Prerequisites
* PyTorch 1.4.0
```bash
pip install torch==1.4.0
```
* NumPy
```bash
pip install numpy
```
* fastText
```bash
pip install fasttext
```
Use the package manager [pip](https://pypi.org/project/genn) to install genn.

## Usage

```python
from genn import Preprocessing, LSTMGenerator
ds = Preprocessing("file.txt")
gen = LSTMGenerator(ds, nLayers = 2,
                        batchSize = 16,
                        embSize = 64,
                        lstmSize = 16,
                        epochs = 20)
#Train the model
gen.run()

# Generate 5 documents
for _ in range(5):
    print(gen.generate_document())
```
#### For more examples on how to use Preprocessing, please refer to [this file](https://github.com/FahedSabellioglu/genn/blob/master/preprocessing_examples.md).
#### For more examples on how to use LSTMGenerator and GRUGenerator, please refer to [this file](https://github.com/FahedSabellioglu/genn/blob/master/generator_examples.md).
## Contributing
 Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
## License
Distributed under the MIT License. See [LICENSE](https://github.com/FahedSabellioglu/genn/blob/master/LICENSE.txt) for more information.
