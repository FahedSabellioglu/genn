
# GPT2 Summarizer
### Input Format
GPT2 expects a .txt file with the following format:
```text
source_document1 = summary1
source_document2 = summary2
...
```
where on each line, an equals sign (" = ") separates a source document and its summary. By default, the text separator is "\n", but you can specify the text separator to be anything during initialization .  Please check the example below.

GPT2 requires task designators in order to train properly. In this case, the task designator is the equals sign, as is mentioned in the paper. 

### Initialization Parameters and Training
```python
from genn import GPT2Summarizer
#Initialize the summarizer
#During initialization, you can set:
# - The file path
# - Number of epochs
# - Batch size, 32 by default
# - GPT variant: (small, medium, large), "small" by default
# - Text separator, "\n" by default,
# - Optimizer parameters, {"lr" : 3e-4} by default
# - Scheduler parameters, {"warmup_steps" : 400} by default
summ = GPT2Summarizer("data.txt",
					epochs=5,
					batch_size=8,
					variant = "medium",
					txtSeparator = ";",
					optimParams = {"lr": 1e-3, 
									"amsgrad": True},
					schedParams = {"warmup_steps":100})

#Train the model
summ.run()

```
### Summarize Document
```python
#To create summaries after training
#During summarization, you can set:
# - The number (n) of summaries to create
# - The source document
# - Max len of generated summary. None by default.
#   Use this if model falls into repetition loop. 
#   GPT2 rarely falls into repetition loop if the data is
#   good. But, you can stop it after 50 tokens for example.
# - Sampling method, isNucleus = True by default.
#   If set to False, then top-5 sampling is used.
#   For Nucleus sampling, you can use the parameter "p"
#   to set the confidence threshold. p = 0.5 by default.
#   For top-k sampling, you can use the parameter "k"
#   to set the candidate pool size. k = 5 by default.
# - No repetition, False by defualt.
#   If the model repeats the same word occasionally,
#   set noRepetition = True. 
src_doc = "This is the source document to summarize"

#Default example
print(summ.summarize_document(n=1, source = src_doc))

#Custom Nucleus Sampling
#with no repetition and max length of 50. 
print(summ.summarize_document(n=1,
			source = src_doc,
			isNucleus = True,
			p = 0.8))
			
#Custom Top=k Sampling with k = 7 and no repetition
print(summ.summarize_document(n=1,
			source = src_doc,
			isNucleus = False,
			k = 7,
			noRepetition = True))
			
```
### Save and load weights
##### There are no built-in functions to do so currently. But, saving and loading are still simple:
```python
#Saving
torch.save(summ.model.state_dict(), path)
#Loading
summ.model.load_state_dict(torch.load(path))
```