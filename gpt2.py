from collections import Counter
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from pytorch_transformers import AdamW, WarmupLinearSchedule
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import csv
import json



class GPT2:
	"""
		High level interface for PyTorch's GPT-2 implementation.
		Parameters
		----------
		fileName: string
			The path to the file (txt, json, csv)
		
		taskToken: string
			GPT-2 expects a task name for generation. For example, for movie titles, this can be "Movie: "
			It will be treated as a fill in the blank task.
		
		epochs: int
			The number of cycles over the training data.

		variant: string, default "small"
			The variant of GPT-2 to use. This can be "small", "medium", "large"

		batchSize: int, default 16
			The number of training instances per iteration.

		eos: string, default "<|endoftext|>"
			The end-of-sentence token.

		instanceMxLen: int, default None
			Max length of each document.

		txtSeparator: str, optional, default '\n'   
			When the data file is a txt file, this is the character serparating documents.
			By default, the assumption is that each document is on a separate line.

		csvIndex: int, optional, default None
			When the data file is a csv file, this is the index of the column to parse from.

		jsonKey: str, optional, default None
			When the data file is a csv file, this is the json key to parse from.
			Typically, it is 'body' or 'text'.

		seedParams: dict, optional, default {'N_first': 1, 'minFreq': 5}
			Parameters that will be used while selecting the random seed used to initalize prediction.
				N_first: get the first n_first tokens of each training document.
				minFreq: The min frequency a seed must have before it is included in the pool. 

				By default random seed is enabled. If seedParams = {} then static seed will be applied.

		optimParams: dict, default {"lr" : 3e-4}
			The optimizer paramters.

		schedParams: dict, default {"warmup_steps" : 400}
			The scheduler paramters.


	"""

	def __init__(self, fileName,
						taskToken,
						epochs,
						variant = "small",
						batchSize = 32,
						eos = "<|endoftext|>",
						instanceMxLen = None,
						txtSeparator = '\n',
						csvIndex = None,
						jsonKey = None,
						seedParams = {'N_first': 1, 'minFreq': 5},
						optimParams = {"lr" : 3e-4},
						schedParams = {"warmup_steps" : 400}):

		self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# To be set by the user
		self.csvIndex = csvIndex
		self.jsonKey = jsonKey
		self.txtSeparator = txtSeparator
		self.eos = eos
		self.taskToken = taskToken
		self.optimParams = optimParams
		self.schedParams = schedParams

		self.epochs = self.__checkIntParam(epochs, "epochs")
		self.batchSize = self.__checkIntParam(batchSize, "batchSize")
		self.fileName = self.__checkFileName(fileName)
		self.instanceMxLen = instanceMxLen
		self.seedParams = self.__checkSeedParams(seedParams)
		self.variant = self.__checkVariant(variant)

		self.examples = None
		self.__text = None
		self.model = None
		self.tokenizer = None
		self.optimizer = None
		self.scheduler = None

		self.__readFile()
		self.__get_tokenizer(self.variant)
		print()
		self.__get_model(self.variant)
		self.__get_optimizer()
		self.__get_scheduler()



	def __checkIntParam(self, param, name):
		if sum([type(param) != int, param <= 0]) >= 1:
			raise Exception("Make sure %s is a positive int."%name)
		return param

	def __checkVariant(self, variant):
		avl = ["small", "medium", "large"]
		if variant not in avl:
			raise Exception("Available variants are", " ".join(avl), "got", variant, "instead")

		#in PyTorch implementation, small = gpt2, medium = gpt2-medium etc.
		return "-"+variant if variant != "small" else ""

	def __checkFileName(self, fileName):
		extension = fileName.split(".")[-1]
		
		if extension not in ['txt', 'csv', 'json']:
			raise Exception("Only works with txt, csv, and json files.")

		if extension == 'csv' and self.csvIndex == None:
			raise Exception("Please provide a 'csvIndex' to indicate the column to parse from.")
		elif extension == 'json' and self.jsonKey == None:
			raise Exception("Please provide a 'jsonKey' to indicate json key to parse from.")
		
		self.extension = extension
		return fileName

	def __checkSeedParams(self,params):
		attr_list = ['N_first', 'minFreq']

		if not len(params):
			return {}
		
		elif not all(attr in params for attr in attr_list):
			raise Exception("seedParams should contain the attributes {attrs}".format(attrs = attr_list))

		return params


	def __readFile(self):
		file_obj = open(self.fileName)

		if self.extension == "txt":
			self.__text = file_obj.read().split(self.txtSeparator)
		elif self.extension == "csv":
			reader = csv.reader(file_obj)
			self.__text = [row[self.csvIndex] for row in reader]
		else:
			json_data = json.loads(file_obj.read())
			self.__text = [row[self.jsonKey] for row in json_data]

		if self.instanceMxLen == None:
			self.instanceMxLen = len(max(self.__text, key=len))
		self.examples = [self.taskToken+" "+inst+" "+self.eos for inst in self.__text 
							if len(inst)>0 and len(inst)<self.instanceMxLen]
		self.seeds =  self.__getStartWords()


	def __getStartWords(self): 
		n_first = 1 if 'N_first' not in self.seedParams else self.seedParams['N_first']

		start_words = [' '.join(instance.split()[:n_first])
						   for instance in self.__text
						   if len(instance.split()) > n_first]
		
		freq_start = Counter(start_words)
		if not len(self.seedParams):
			word, _ = freq_start.most_common(1)[0]
			print("The word {w} is the Static Seed. ".format(w=word))
			return {word: 1.0}
		
		return {word:freq/len(self.examples) for word,freq  in freq_start.items() if freq >= self.seedParams['minFreq'] }


	def getSeed(self):
		"""
			return a weighted seed. 
			In case static seed is enabled, then the most frequent token will be the seed.
		"""
		seeds = list(self.seeds.keys())
		probs = list(self.seeds.values())
		return np.random.choice(seeds, 1 , probs).tolist()

	def __get_batches(self):            
		num_batches = len(self.examples) // self.batchSize
		for i in range(0, num_batches*self.batchSize, self.batchSize):
			yield self.examples[i:i+self.batchSize]

	def __encode_batch(self, batch):
		encoded = torch.Tensor().long().to(self.device)
		for inst in batch:
			docTens = torch.tensor(self.tokenizer.encode(inst)).unsqueeze(0).to(self.device)
			encoded = torch.cat([encoded, docTens[:,1:]], dim=1)
		return encoded


	def __get_optimizer(self):
		self.optimizer = AdamW(self.model.parameters(), **self.optimParams)

	def __get_model(self, variant):
		self.model = GPT2LMHeadModel.from_pretrained('gpt2'+variant)
		self.model = self.model.to(self.device)
		self.model.train()

	def __get_tokenizer(self, variant):
		self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2'+variant)

	def __get_scheduler(self):
		self.scheduler = WarmupLinearSchedule(self.optimizer, t_total = -1, **self.schedParams)

	def choose_from_top(self, probs, n=5):
		ind = np.argpartition(probs, -n)[-n:]
		top_prob = probs[ind]
		top_prob = top_prob / np.sum(top_prob) # Normalize
		choice = np.random.choice(n, 1, p = top_prob)
		token_id = ind[choice][0]
		return int(token_id)

	def select_nucleus(self, out, p = 0.5):
		probs = F.softmax(out, dim=-1)
		idxs = torch.argsort(probs, descending = True)
		res, prob,  cumsum = [], [], 0.
		for idx in idxs:
			res.append(idx)
			prob.append(probs[idx].item())
			cumsum+=probs[idx]
			if cumsum>p:
			   break
		nucleus_prob = prob / np.sum(prob)
		choice = np.random.choice(res , 1,p = nucleus_prob)[0]
		return choice


	def run(self):
		sum_loss = 0.0
		batch_count = 0
		last_loss = None
		batches_len = (len(self.examples) // self.batchSize) * self.epochs
		for e in range(self.epochs):
			batches = self.__get_batches()
			batch_times = []
			for batch in batches:
				start_time = time.time()

				encoded = self.__encode_batch(batch)

				outputs = self.model(encoded, labels=encoded)

				loss, logits = outputs[:2]                        
				loss.backward()
				sum_loss += loss.detach().data

				batch_count += 1
				self.optimizer.step()
				self.scheduler.step() 
				self.optimizer.zero_grad()
				self.model.zero_grad()


				batch_time = time.time() - start_time
				batch_times.append(batch_time)

				mean_time = sum(batch_times)/len(batch_times)
				remaining_batches = batches_len - batch_count       
				remaining_seconds = remaining_batches * mean_time 
				remaining_time = time.strftime("%H:%M:%S",
					time.gmtime(remaining_seconds))
				progress = "{:.2%}".format(batch_count/batches_len)
				print('Epoch: {}/{}'.format(e+1, self.epochs),
					  'Progress:', progress,
					  'Loss: {}'.format(last_loss),
					  'ETA:', remaining_time)
			last_loss = sum_loss
			sum_loss = 0.0


	def generate_document(self, n, isNucleus=True, instanceMxLen=None, k=None, p=None, uniq=True):
		self.model.eval()
		res = set()
		max_len = instanceMxLen if instanceMxLen!=None else self.instanceMxLen
		with torch.no_grad():
			while len(res) < n:
				cur_ids = torch.tensor(self.tokenizer.encode(self.taskToken+" "+self.getSeed()[0])).unsqueeze(0).to(self.device)

				for i in range(max_len):
					outputs = self.model(cur_ids, labels=cur_ids)
					_, logits = outputs[:2]

					if isNucleus:
						if p!=None:
							next_token_id = self.select_nucleus(logits[0,-1], p=p)
						next_token_id = self.select_nucleus(logits[0,-1])

					else: #topk
						softmax_logits = torch.softmax(logits[0,-1], dim=0)
						if k!=None:
							next_token_id = self.choose_from_top(softmax_logits.to('cpu').numpy(), n=k)
						next_token_id = self.choose_from_top(softmax_logits.to('cpu').numpy())


					cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(self.device) * next_token_id], dim = 1)
					if next_token_id in self.tokenizer.encode(self.eos):
						break

				doc = self.tokenizer.decode(list(cur_ids.squeeze().to('cpu').numpy())).strip()
				if uniq:
					if doc not in self.examples:
						res.add(doc)
				else:
					res.add(doc)

		return list(res)



