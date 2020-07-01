
# GPT2
### Default
```python
from genn import GPT2


#Examples for creating GPT-2 generators:
#Creating a default model.
gen = GPT2("data.txt",
            taskToken = "Movie:",
            epochs = 10)
#Use help() for full parameter list.
help(GPT2)

```
### User defined parameters
```python
#Exmaple with different parameters
from genn import GPT2
gen = GPT2('data.csv',
            taskToken = "Movie:",
            epochs = 5,
            variant = "medium",
            batchSize = 64,
            eos = "!EOS!",
            instanceMxLen = 140,
            csvIndex = 2,
            seedParams = {'minFreq': 5},
            optimParams = {"lr" : 6e-3},
            schedParams = {"warmup_steps" : 200})


```
# Functions
```python
# Training the GPT-2 model
gen.run()

# Generating a document
doc = gen.generate_document(1)

# In documents generation default prediction parameters can be overwritten.
# Overwrite the selection method to Top-k selection with k=6

docs = gen.generate_document(10, isNucleus = False, k = 6)
# Overwrite nucleus threshold p to 0.8, instead of 0.5. Generate 40 documents
docs = gen.generate_document(40, p = 0.8)
