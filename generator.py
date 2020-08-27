import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from genn.embedding import PretrainedEmbeddings
from genn.preprocessing import Preprocessing
from genn.controllers import checkDropout, checkParams, checkSeedParams, checkProb, selectNucleus, chooseFromTop

from abc import abstractmethod, ABCMeta

class Generator(nn.Module, metaclass=ABCMeta):
    

    @checkParams(Preprocessing, int, int, int, int, int, (float, int), str, int, (str, type(None)), dict,
                     (str, type(None)), (str, type(None)), bool, dict)
    def __init__(self, preproObj, nLayers, batchSize, embSize, rnnSize, epochs, dropout,
                        embeddingOption='fasttext', genIteration = 10, glovePath = None,
                        fastTextParams = {}, loadFastText = None, saveFastText = None,
                        fineTuneEmbs = False, selectionParams = {"sType": 'topk', 'k': 5, 'probThreshold': 0.5}
                        ):
        super(Generator, self).__init__()

        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.preproObj = preproObj
        self.__vocabToInt = preproObj.getVocab.vocab.stoi
        self.__intToVocab = preproObj.getVocab.vocab.itos

        self.__genIter = genIteration
        self.batchSize = batchSize
        self.embSize = embSize
        self.nLayers = nLayers
        self.epochs = epochs
        self.rnnSize = rnnSize
        self.dropout = checkDropout(dropout, nLayers)
        self.nVocab = len(preproObj.getVocab.vocab)

        self.embedding = self.__getEmbs(embeddingOption, fastTextParams, loadFastText, saveFastText, glovePath, fineTuneEmbs)
        
        self.__selection = None
        self.__topk = 5
        self.__nucluesProb = 0.5
        self.__checkSelection(selectionParams)

    def __checkSelection(self,value):
        if not len(value):
            raise Exception("The dict selectionParams cannot be empty")

        if 'sType' not in value:
            raise Exception("The dictionary has to have the key sType that indicates the selection type")
        
        else:
            selection = value['sType'].lower()
            if selection in 'topk':
                self.__selection = 'topk'
                if 'k' in value:
                    self.__topk = value['k']

            elif selection in 'nucleus':
                self.__selection = 'nucleus'
                if 'probThreshold' in value:
                    self.__nucluesProb = checkProb(value['probThreshold'])
            else:
                raise Exception("sType can only be topk or nucleus")

    def __checkEmbedding(self,value):
        value = value.lower()
        if value in 'glove': 
            return 'glove'

        elif value in 'fasttext':
            return 'fasttext'

        elif value in 'random':
            return 'random'

        raise Exception("Embedding type can only be glove, fasttext or random(nn.Embedding)")

    def __getEmbs(self, embOption, fastTextParams, loadFastText, saveFastText, glovePath, fineTuneEmbs):
        self.__embOption = self.__checkEmbedding(embOption)
        self.__fineTuneEmbs = fineTuneEmbs


        if self.__embOption in ['glove', 'fasttext']:

            pretrainedEmbeddings = PretrainedEmbeddings(self.preproObj,self.embSize ,self.__embOption,
                glovePath, fastTextParams, loadFastText, saveFastText)
            
            weights = pretrainedEmbeddings.weights
            
            self.__checkembedDims(weights.shape[1])
                        
            return self.__createEmbLayer(weights, nonTrainable=fineTuneEmbs)

        else:
            return nn.Embedding(self.nVocab, self.embSize, padding_idx=0)

    def __checkembedDims(self, value):
        if self.embSize != value:
            raise Exception("The pretrained embedding size does not equal the input size that the rnn will get\n"
                            f"pretrained embedding size = {value} , LSTM input = {self.embSize}")
        return value

    def __createEmbLayer(self, weightsMatrix, nonTrainable=False):
        numEmbeddings, embeddingDim = weightsMatrix.shape
        embLayer = nn.Embedding(numEmbeddings, embeddingDim,padding_idx=0)
        embLayer.weight.data.copy_(torch.from_numpy(weightsMatrix))
        if nonTrainable:
            embLayer.weight.requires_grad = False

        return embLayer
    
    def getBatches(self):
        numBatches = len(self.preproObj) // self.batchSize
        for i in range(0, numBatches * self.batchSize, self.batchSize):
            yield self.preproObj[i:i+self.batchSize]

    def packSrc(self, embed,length):
        return rnn_utils.pack_padded_sequence(embed, length, batch_first=True)
    
    def padPack(self, packedOutput):
        return rnn_utils.pad_packed_sequence(packedOutput, padding_value=1)
    
    def padSeq(self, batch):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=1).long()

    def getLossAndTrainOp(self, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return criterion, optimizer

    def getSrcTrg(self, batch):
        """
         batches shape: batch_size * tokens
        """        
        srcBatch,srcLength = self.prepareBatch(batch)
        trgBatch,trgLength = self.prepareBatch(batch, IsSrc=False)
        
        return srcLength, trgLength, srcBatch, trgBatch

    def prepareBatch(self, exampleObjects, IsSrc=True):
        """
         turn the tokens into padded seqs and get the length of each seq separately.
        """
        targetToken, targetLength = ('src_token','src_length') if IsSrc else ('trg_token','trg_length')
            
        batch = self.padSeq([torch.Tensor(getattr(exp,targetToken)) for exp in exampleObjects])
        
        lengths = torch.tensor([getattr(exp,targetLength) for exp in exampleObjects])
        
        lengths , permIdx = lengths.sort(0, descending=True)
        
        batch = batch[permIdx]
        
        return batch, lengths

    def zeroState(self, n, batchSize):
        return ([torch.zeros(self.nLayers, batchSize, self.rnnSize)]) * n  

    @checkParams(str)
    def save(self, path):
        fileName = f"{type(self).__name__}-{self.__embOption}-{self.embSize}.pth"
        
        if not os.path.exists(path):
            os.mkdir(path)

        torch.save({
            'nLayers': self.nLayers,
            'embSize': self.embSize,
            'rnnSize': self.rnnSize,
            'dropout': self.dropout,
            'model_state_dict': self.state_dict(),
            }, os.path.join(path, fileName))

    @checkParams(str)
    def load(self, path):
        savedModel = torch.load(path)
        modelStateDict = savedModel["model_state_dict"]
        nLayers = savedModel["nLayers"]
        embSize = savedModel["embSize"]
        rnnSize = savedModel["rnnSize"]
        if sum([self.nLayers!=nLayers, 
                self.embSize!=embSize,
                self.rnnSize!=rnnSize]) > 0:
            raise Exception("Please create a model with the same parameters as the trained one. Trained model parameters: {} nLayers, {} embSize, {} lstmSize/gruSize".format(nLayers,embSize,rnnSize))

        self.load_state_dict(modelStateDict)

    @abstractmethod
    def run(self):
        pass
    @abstractmethod
    def forward(self, x, length, prevState):
        pass

    @checkParams(int, (int, type(None)), (str, type(None)),(int, type(None)), (float, type(None)), bool )
    def generateDocument(self, n, genIter = None, selection = None, k = None, prob = None, uniq=True):
        """
            Generate documents.
            selection: The type of selection will be used. its either topk or nucleus, t and n can also be used.
            k: In case of topk, it represents the k value.
            prob: In case of nucleus, its represnts the probability thresholds 
            predIter: The number of iterations for the prediction loop unless the model generated a stop token.


            They are optional while calling this method. The values passed while creating the generator object will be used.
        """
        selection = (selection or self.__selection)
        k = (k or self.__topk)
        prob = (prob or self.__nucluesProb)
        genIter = (genIter or self.__genIter)
        eos =  self.__vocabToInt[self.preproObj.getVocab.eos_token]
        modelName = type(self).__name__

        self.eval()
        res = set()
        while len(res) < n:
            words = self.preproObj.getSeed()
            choice = self.__vocabToInt[words[0]]

            if 'GRU' in modelName:
                states = self.zeroState(1, 1)[0]        
            else:
                states = self.zeroState(2, 1)
            with torch.no_grad():
                for _ in range(genIter):
                    ix = torch.tensor([[choice]]).long().to(self.device)
                    length = torch.tensor([ix.shape[1]])    

                    output, states = self(ix,length,states)

                    if selection in 'nucleus':
                        choice = selectNucleus(output.view(1,-1),p=prob)

                    elif selection in 'topk':
                        choice = chooseFromTop(output.view(1,-1),n=k)
                        
                    if choice == eos:
                        words.append(self.__intToVocab[eos])
                        break                            
                    words.append(self.__intToVocab[choice])

                doc = ' '.join(words)
                if uniq:
                    if doc not in self.preproObj.examples:
                        res.add(doc)
                else:
                    res.add(doc)
        return list(res)



    def info(self):
        """
            Show the parameters used with the generator.
        """
        generatorType = type(self).__name__
        print(self.embedding)
        print(f"Size of data: {len(self.preproObj)} documents.")
        print("Device used:", self.device)
        print("Vocabulary size:", self.nVocab)
        print(f"Layers of {generatorType}:", self.nLayers)
        print("Dropout:", f"True, {self.dropout}" if self.dropout>0 else False)
        print(f"{generatorType} size:", self.rnnSize)
        print("Batch size:", self.batchSize)
        print("Embedding type:", self.__embOption)
        print("Fine tune embeddings?:", self.__fineTuneEmbs)
        print("Embedding size:", self.embSize)
        print("Selection method:", f"{self.__selection}, k = {self.__topk}" if self.__selection == "topk"
            else f"{self.__selection} p = {self.__nucluesProb}")


