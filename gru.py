import time

import torch
import torch.nn as nn

from genn.controllers import timeChecker
from genn.generator import Generator
from genn.preprocessing import Preprocessing


class GRUGenerator(Generator):
	"""
	GRU arch for text generation

	Parameters
	----------
	datasetObj : Preprocessing class object
		An object created using the class Preprocessing.

	nLayers: int
		The number of GRU layers

	batchSize: int
		The number of training instances per iteration.

	embSize: int
		The size of the vector in which each token will be embedded.

	gruSize: int
		The number of features in the hidden state.

	epochs: int
		The number of cycles over the training data

	dropout: float, optional, default 0
		The percentage of units that will be dropped per layer.

	embeddingOption: string, optional, default random
		The type of embeddings that will be used for the tokens.
		It can be glove, fasttext or random. Random means the NN.Embedding will be used.

	predictionIteration: int, optional, default 10
		The number of iterations for the prediction loop unless the model generates the stop token.

	glovePath: string, optional, default None
		The path for the GloVE embeddings if embeddingType is set to 'glove'.

	fastTextParams: dict, optional, default {}
		The parameters will be used while fine-tuning fastText to create the embeddings.
		Emtpy dictionary means the default training parameters will be used.

	loadFastText: string, optional, default None
		A path to the pretrained fastText model file.

	saveFastText: string, optional, default None
		A path to save the fastText model after training.

	fineTuneEmbs: bool, optional,  default False
		Whether to fine-tune the pretrained embeddings or not.

	selectionParams: dict, optional, default {"sType": 'topk', 'k': 5}
		sType: The type of selection method to use during prediciton.
			It is either 'topk' or 'nucleus', 't' and 'n' can also be used.
		k: in case of topk, it represents the k value.
		probThreshold: When using Nucleus selection, this represnts the probability threshold p.
	"""

	def __init__(
				self,preproObj, nLayers, batchSize, embSize, gruSize,epochs,
				dropout = 0.0 , embeddingOption ='fasttext',genIteration = 10, glovePath = None,
				fastTextParams = {},loadFastText = None, saveFastText = None, fineTuneEmbs = False,
				selectionParams = {"sType": 'topk', 'k': 5, 'probThreshold': 0.5}):


		super(GRUGenerator, self).__init__(preproObj, nLayers, batchSize, embSize,
										gruSize, epochs, dropout, embeddingOption, genIteration,
										glovePath, fastTextParams, loadFastText, saveFastText, fineTuneEmbs,
										selectionParams)

		self.gru = nn.GRU(self.embSize,
						self.rnnSize,
						num_layers = self.nLayers,
						dropout = self.dropout,
						batch_first = True)
		self.dense = nn.Linear(self.rnnSize, self.nVocab)

	def forward(self, x, length, prevState):
		"""
			params:
				x: the tensor of input instances.
				length: the length that will be used from each training instance to handle paddings.
				prev_state: cell state.

		"""
		embed = self.embedding(x)
		packedInput = self.packSrc(embed, length)
		packedOutput, state = self.gru(packedInput, prevState)
		padded, _ = self.padPack(packedOutput)
		logits = self.dense(padded)
		return logits, state

	def run(self):
		"""
			Training method
		"""
		self.to(self.device)
		criterion, optimizer = self.getLossAndTrainOp(0.01)

		batchCount = 0
		batchesLen = (len(self.preproObj) // self.batchSize) * self.epochs

		for e in range(self.epochs):

			batches = self.getBatches()

			stateH = self.zeroState(1, self.batchSize)[0]
			stateH = stateH.to(self.device)

			batchTimes = []

			for batch in batches:

				startTime = time.time()
				srcLengths, _ ,srcBatch,trgBatch = self.getSrcTrg(batch)

				batchCount += 1

				self.train()
				optimizer.zero_grad()

				trgBatch = torch.t(trgBatch).to(self.device)
				srcBatch = srcBatch.to(self.device)

				logits, stateH = self(srcBatch, srcLengths, stateH)

				loss = criterion(logits.transpose(1, 2), trgBatch)

				stateH = stateH.detach()

				lossValue = loss.item()

				loss.backward(retain_graph=True)

				optimizer.step()

				loss.backward()

				_ = nn.utils.clip_grad_norm_(
				self.parameters(), 5)

				optimizer.step()

				batchTime = time.time() - startTime
				batchTimes.append(batchTime)

				if batchCount % 10 == 0:
					timeChecker(batchTimes, batchCount, batchesLen, e, self.epochs, lossValue)
