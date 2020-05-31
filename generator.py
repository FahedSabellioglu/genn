import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from random import random
import numpy as np

from embedding import PretrainedEmbeddings
from preprocessing import Preprocessing


class Generator(nn.Module):

    def __init__(self,datasetObj, nLayers, batchSize, embSize, rnnSize,
                   epochs , dropout , embeddingType ='fasttext', predictionIteration = 10,
                   glovePath = None, fastTextParams = {},
                   loadFastText = None, saveFastText = None, fineTuneEmbs = False, 
                   selectionParams = {"sType": 'topk', 'k': 5, 'probThreshold': 0.5}):
        super(Generator, self).__init__()

        self.db = self.__checkDataset(datasetObj)

        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vocabToInt = self.db.DataVocab.vocab.stoi
        self.intToVocab = self.db.DataVocab.vocab.itos

        self.predIter = self.__checkIntParam(predictionIteration, 'predictionIteration')
        self.batchSize = self.__checkIntParam(batchSize,'batchSize')
        self.embSize = self.__checkIntParam(embSize, "embSize")
        self.nLayers = self.__checkIntParam(nLayers, "nLayers")
        self.epochs = self.__checkIntParam(epochs, "epochs")
        self.rnnSize = self.__checkIntParam(rnnSize, "lstm_size/gru_size")
        self.fineTuneEmbs = fineTuneEmbs


        self.glovePath = self.__checkPath(glovePath, "glovePath")
        self.loadFastText = self.__checkPath(loadFastText, "loadFastText")
        self.saveFastText = self.__checkPath(saveFastText, "saveFastText")

        self.dropout = self.__checkDropout(dropout)
        self.embeddingType = self.__checkEmbedding(embeddingType)
        self.fastTextParams = self.__checkDictParams(fastTextParams)

        self.n_vocab = len(self.db.DataVocab.vocab)
        self.embedding = self.__getEmbs()
        
        self.selection = None
        self.topk = 5
        self.nucluesProb = 0.5
        self.__checkSelection(selectionParams)

    def __checkSelection(self,value):
        self.__checkDictParams(value)
        if not len(value):
            raise Exception("The parameter cannot be empty")

        if 'sType' not in value:
            raise Exception("The dictionary has to have the key sType that indicates the selection type")
        
        else:
            selection = value['sType'].lower()
            if selection in 'topk':
                self.selection = 'topk'
                if 'of' in value:
                    self.topk = value['k']
                else:
                    print("Setting topk to 5 by default")

            elif selection in 'nucleus':
                self.selection = 'nucleus'
                if 'probThreshold' in value:
                    self.nucluesProb = self.__checkProb(value['probThreshold'])
                else:
                    print("Setting nuclues threshold to 0.5 by default")
            else:
                raise Exception("sType can only be topk or nucleus")

    def __checkProb(self,value):
        if sum([type(value) == float, 0 <= value <= 1]) != 2:
            raise Exception("Make sure the probThreshold is a positive float, between 0 and 1." )
        return value

    def __checkDataset(self, dataset):
        filetype = type(dataset).__base__.__name__
        if filetype != "Dataset":
            raise Exception("Please provide a Dataset object. Instead, got '%s'."% filetype)
        return dataset
    
    def __checkEmbedding(self,value):
        value = value.lower()
        if value in 'glove': # both glove and g will be accepted.
            return 'glove'
        elif value in 'fasttext':
            return 'fasttext'

        raise Exception("Embedding type can only be glove or fasttext.")

    def __getEmbs(self):
        if self.embeddingType in ['glove', 'fasttext']:
            pretrainedEmbeddings = PretrainedEmbeddings(self.db,self.embSize ,self.embeddingType,
                self.glovePath, self.fastTextParams, self.loadFastText, self.saveFastText)
            
            weights = pretrainedEmbeddings.weights
            
            self.__checkembedDims(weights.shape[1])
                        
            return self.create_emb_layer(weights, non_trainable=self.fineTuneEmbs)

        else:
            return nn.Embedding(self.n_vocab, self.embSize, padding_idx=0)
    
    def __checkembedDims(self, value):
        if self.embSize != value:
            raise Exception("The pretrained embedding size does not equal the input size that the rnn will get\n"
                            f"pretrained embedding size = {value} , LSTM input = {self.embSize}")
        return value

    def create_emb_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim,padding_idx=0)
        emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer
    
    def __checkIntParam(self, param, name):
        if sum([type(param) != int, param <= 0]) >= 1:
            raise Exception("Make sure %s is a positive int."%name)
        return param

    def __checkDropout(self, dropout):
        if self.nLayers < 2 and dropout != 0:
            raise Exception("The network must have more than 1 layer to have dropout.")
        if dropout < 0 or dropout > 1:
            raise Exception("Dropout value must be between 0 and 1.")     
        return dropout   

    def __checkDictParams(self, value):
        if type(value) != dict:
            raise Exception("Please provide a proper dict containing the parameters.")
        return value
    
    def __checkPath(self, param, name):
        if not param : return None
        elif type(param) != str:
            raise Exception("Please provide a proper path for %s"%name)
        return param

    def get_batches(self):
        num_batches = len(self.db) // self.batchSize
        for i in range(0, num_batches * self.batchSize, self.batchSize):
            yield self.db[i:i+self.batchSize]

    def pack_src(self, embed,length):
        return rnn_utils.pack_padded_sequence(embed, length, batch_first=True)
    
    def pad_pack(self, packed_output):
        return rnn_utils.pad_packed_sequence(packed_output)
    
    def pad_seq(self, batch):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True).long()

    def get_loss_and_train_op(self, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return criterion, optimizer

    def get_src_trg(self, batch):
        """
         batches shape: batch_size * tokens
        """        
        src_batch,src_length = self.prepare_batch(batch)
        trg_batch,trg_length = self.prepare_batch(batch, IsSrc=False)
        
        return src_length, trg_length, src_batch, trg_batch

    def prepare_batch(self, example_objects,IsSrc=True):
        """
         turn the tokens into padded seqs and get the length of each seq separately.
        """
        target_token,target_length = ('src_token','src_length') if IsSrc else ('trg_token','trg_length')
            
        batch = self.pad_seq([torch.Tensor(getattr(exp,target_token)) for exp in example_objects])
        
        lengths = torch.tensor([getattr(exp,target_length) for exp in example_objects])
        
        lengths , perm_idx = lengths.sort(0, descending=True)
        
        batch = batch[perm_idx]
        
        return batch, lengths

    def zero_state(self, n, batchSize):
        return ([torch.zeros(self.nLayers, batchSize, self.rnnSize)]) * n      

    def select_nucleus(self, out, p = None):
        p = (p or self.nucluesProb)
        probs = F.softmax(out, dim=-1)[0]
        idxs = torch.argsort(probs, descending = True)
        res, cumsum = [], 0.
        for idx in idxs:
            res.append(idx)
            cumsum+=probs[idx]
            if cumsum > p: return res
    
    def info(self, generatorType):
        print(f"Layers of {generatorType}:", self.nLayers)
        print("Batch size:", self.batchSize)
        print("Embedding size:", self.embSize)
        print(f"{generatorType} size:", self.rnnSize)
        print("Using Nucleus prediction?:", True if self.selection =='Nucleus' else False )
        print("Dropout:", (True, self.dropout) if self.dropout>0 else False)


    def save(self, path):
        directory = "/".join(path.split('/')[:-1])
        if not os.path.exists(directory):
            os.mkdir(directory)

        torch.save({
            'nLayers': self.nLayers,
            'embSize': self.embSize,
            'rnnSize': self.rnnSize,
            'dropout': self.dropout,
            'model_state_dict': self.state_dict(),
            }, path)

    def load(self, path):
        saved_model = torch.load(path)
        model_state_dict = saved_model["model_state_dict"]
        nLayers = saved_model["nLayers"]
        embSize = saved_model["embSize"]
        rnnSize = saved_model["rnnSize"]
        if sum([self.nLayers!=nLayers, 
                self.embSize!=embSize,
                self.rnnSize!=rnnSize]) > 0:
            raise Exception("Please create a model with the same parameters as the trained one. Trained model parameters: {} nLayers, {} embSize, {} lstmSize/gruSize".format(nLayers,embSize,rnnSize))

        self.load_state_dict(model_state_dict)

    
    def generateDocument(self, modelName, predIter = None, selection = None, k = None, prob = None):
        selection = (selection or self.selection)
        k = (k or self.topk)
        prob = (prob or self.nucluesProb)
        predIter = (predIter or self.predIter)

        self.eval()

        words = self.db.getSeed()
        choice = self.vocabToInt[words[0]]

        if modelName == "GRU":
            states = self.zero_state(1, 1)[0]
        
        else:
            states = self.zero_state(2, 1)
        
        for _ in range(predIter):
            ix = torch.tensor([[choice]]).long().to(self.device)
            length = torch.tensor([ix.shape[1]])

            output, states = self(ix,length,states)

            if selection == 'nucleus':
                choices = self.select_nucleus(output.view(1,-1), p= prob)

            elif selection == 'topk':
                _, top_ix = torch.topk(output[0], k=k)
                choices = top_ix.tolist()[0]
            
            choice = np.random.choice(choices)

            
            eos =  self.vocabToInt["."]
            if eos == max(choices) and 0.7 > random():            
                choice = eos
                
            if choice == eos:
                words.append(self.intToVocab[choice])
                return words
                    
            words.append(self.intToVocab[choice])

        return(words)



        





    
