# Preprocessing
### Default
```python
from genn import Preprocessing
# Create a dataset object using Preprocessing.
#     To do so, initialize Preprocessing with your file.
#     Accepted formats: txt, json, and csv.
# By default:
#     -The text is tokenized using regex.
#     -Instances documents of all lengths are included.
#     -The tokens become lower case.
#     -The token <!EOS!> is appended to the end of each
#      document to indicate end-of-sentence.
#     -Random seed is created as such:
#      -Sample the first N = 1 tokens from all instances.
#      -Ignore tokens that appear less than minFreq = 5 times.


#Parsing examples:
# Read from the CSV file where documents are in column 1.
ds = Preprocessing('data.csv', csvIndex=1)

# Read from txt file where documents are separated by 2 newlines.
# Default txtSeparator = "\n".
ds = Preprocessing('data.txt', txtSeparator = "\n\n")

# Read from JSON file where documents are in JSON key "body".
ds = Preprocessing('data.json', jsonKey='body')
```
### User defined params
```python
from genn import Preprocessing

# Disable dynamic seed and use the most frequent word as a seed.
ds = Preprocessing('data.txt', seedParams={})

# Do not include documents that have more than 20 tokens.
ds = Preprocessing('data.txt', instanceMxLen=20)

# Tokenize using the word_tokenize() function from nltk
ds = Preprocessing('data.txt', tokenizationMethod='nltk')

# Create a Field object with custom params.
# You can find all available params by
# checking the documentation of torchtext.data.field
ds = Preprocessing('data.txt', fieldParams = {'lower':True,
                                              'eos_token': "<EOS>",
                                              'include_lengths': True}
# All the above is compatible with CSV and JSON files as well.
```
### Accessible attributes and functions
```
# print the field object
print(ds.DataVocab)

# print stoi dictionary
print(ds.DataVocab.vocab.stoi)

# print itos list
print(ds.DataVocab.itos)

# get a random seed
print(ds.getSeed())
```
