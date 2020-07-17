# GRUGenerator and LSTMGenerator
### Default
```python
from genn import GRUGenerator, LSTMGenerator, Preprocessing

ds = Preprocessing('data.txt')
# To create an LSTM or GRU generator, provide the
# dataset object created by Preprocessing. 
# By default:
#     -Dropout = 0
#     -Embedding type = 'fasttext'
#     -predictionIteration = 10
#      (Number of tokens to generate when 
#       calling generate_document() unless
#       eos token is predicted)
#     -No embedding fine-tuning.
#     -Next token selection method:
#      Top-k selection, with k = 5.

#Examples for creating generators:
# Using GRU architecture
gen = GRUGenerator(ds, nLayers = 1,
                        batchSize = 64,
                        embSize = 32,
                        gruSize = 32,
                        epochs = 7)

# Using LSTM architecture
gen = LSTMGenerator(ds, nLayers = 1,
                         batchSize = 64,
                         embSize = 32,
                         lstmSize = 32,
                         epochs = 7)
```
### User defined params
```python
from genn import GRUGenerator, LSTMGenerator, Preprocessing
ds = Preprocessing('data.txt')
# Load a pre-trained fasttext model to be used for the embedding.
# Use Nucleus selection with probability threshold 0.8.
gen = GRUGenerator(ds, nLayers = 1,
                        batchSize = 64,
                        embSize = 32,
                        gruSize = 32,
                        epochs = 7,
                        loadFastText='fastText_64.bin',
                        selectionParams = {'sType':'n', 'probThreshold': 0.8})

# Train a fasttext model for the embedding and save the model in the given path.
# Fine tune the embedding vectors.
gen = LSTMGenerator(ds, nLayers = 1,
                         batchSize = 64,
                         embSize = 32,
                         lstmSize = 32,
                         epochs = 7,
                         saveFastText = 'foo_bar.bin',
                         fineTuneEmbs = True)

# In this example, GloVe embeddings are used and will overwrite the parameter embSize.
# Generate up to 20 tokens in a single instance
gen = GRUGenerator(ds, nLayers = 1,
                        batchSize = 64,
                        embSize = 64,
                        gruSize = 32,
                        epochs = 7,
                        embeddingType ='glove',
                        glovePath ='glove.6B.50d',
                        predictionIteration = 20,
                        selectionParams = {'sType':'n', 'probThreshold': 0.8})
```
# Functions
```python
# Training the model whether the model is GRU or LSTM
gen.run()

# Save the trained model
gen.save("models/best_model.pt")

# If you already trained and saved a model, just
#  load it instead of training again.
# Make sure the gen is initialized with the same 
# nLayers, embSize, and gruSize/lstmSize, and dropout.
# If you forgot the model paramters, just load and the
#  error will inform you.
gen.load("models/best_model.pt")

# Generating 5 documents
gen.generate_document(5)

# In documents generation default prediction parameters can be overwritten.
# Overwrite the selection method to Nucleus selection with a probability of 0.7
# Generate up to 40 tokens per document for 10 documents
gen.generate_document(10, predIter = 40, selection = 'n', prob = 0.7)
