import torch
import torch.nn as nn

from generator import Generator
from preprocessing import Preprocessing

from sklearn.feature_extraction import FeatureHasher

class GRUGenerator(Generator):
    """
        GRU arch for text generation
        Parameters
        ----------
        datasetObj : Preprocessing class object
            An object created using the class Preprocessing that has the paris of src and trg documents
            tokenized and shifted

        nLayers: int
            The number of recurrent layers
        
        batchSize: int
            The number of training instances per iteration

        embSize: int
            The size of the vector space which the tokens will be embedded
        
        rnnSize: int
            The number of features in the hidden state
        
        epochs: int
            The number of cycles over the training data
        
        dropout: int, optional, default 0
            The percentage of units that will be dropped per layer.
        
        embeddingType: string, optional, default fasttext
            The type of embeddings that will be used for the tokens.
            It can be glove, fasttext or default. default means the NN.Embedding will be used.
        
        predictionIteration: int, optional, default 11
            The number of iterations for the prediction loop unless the model generated a stop token.
        
        glovePath: string, optional, default None
            The path for the glove embeddings if embeddingType is set to glove.
        
        fastTextParams: dict, optional, default {}
            The parameters will be used while training the fasttext to create the embeddings.
            Emtpy dictionary means the default training parameters will be used.
        
        loadFastText: string, optional, default None
            A path to load pretrained fasttext model.
        
        saveFastText: string, optional, default None
            A path to save the fasttext model after training.
        
        fineTuneEmbs: bool, optional,  default False
            Whether to fine tune the pretrained embeddings or not.

        selectionParams: dict, optional, default {"sType": 'topk', 'k': 5, 'probThreshold': 0.5}
            sType: the type of selection will be used. its either topk or nucleus, t and n can also be used.
            k: in case of topk, it represents the k value.
            probThreshold: in case of nucleus, its represnts the probability thresholds 

    """



    def __init__(self,datasetObj, nLayers, batchSize, embSize, rnnSize,
                   epochs , dropout = 0 , embeddingType ='fasttext' ,
                   predictionIteration = 11 ,glovePath = None, fastTextParams = {},
                   loadFastText = None, saveFastText = None, fineTuneEmbs = False, 
                   selectionParams = {"sType": 'topk', 'k': 5, 'probThreshold': 0.5}):

        super(GRUGenerator, self).__init__(datasetObj,nLayers,batchSize,embSize,
                                           rnnSize,epochs,dropout,embeddingType,predictionIteration,
                                           glovePath,fastTextParams,loadFastText,saveFastText,fineTuneEmbs,
                                           selectionParams)

        self.gru_size = self.rnnSize

        self.gru = nn.GRU(self.embSize,
                        self.gru_size,
                        num_layers = self.nLayers,
                        dropout = self.dropout,
                        batch_first = True)
        self.dense = nn.Linear(self.gru_size, self.n_vocab)

  

    def forward(self, x, length, prev_state):
        """
            params:
                x: a tensor of input instances.
                length: the length that will be used from each training instance to handle paddings.
                prev_state: Cell state

        """
        embed = self.embedding(x)
        packed_input = self.pack_src(embed, length)
        packed_output, state = self.gru(packed_input, prev_state)
        padded,_ = self.pad_pack(packed_output) 
        logits = self.dense(padded)
        return logits, state



    def info(self):
        """
            Show the parameters used with the generator.
        """
        super(GRUGenerator, self).info('GRU')


    def run(self):
        """
            Training method
        """
        criterion, optimizer = self.get_loss_and_train_op(0.01)
        iteration = 0
        for e in range(self.epochs):        
            batches = self.get_batches()
            
            state_h = self.zero_state(1, self.batchSize)[0]
            state_h = state_h.to(self.device)
            
            for batch in batches:
                src_lengths, _ ,src_batch,trg_batch = self.get_src_trg(batch)
                
                iteration += 1

                self.train()
                optimizer.zero_grad()
                
                trg_batch = torch.t(trg_batch).to(self.device) 
                src_batch = src_batch.to(self.device)

                logits, state_h = self(src_batch, src_lengths, state_h)

                loss = criterion(logits.transpose(1, 2), trg_batch)

                state_h = state_h.detach()

                loss_value = loss.item()

                loss.backward(retain_graph=True)

                optimizer.step()

                loss.backward()

                _ = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 5)

                optimizer.step()

                if iteration % 100 == 0:                        
                    print('Epoch: {}/{}'.format(e, self.epochs),
                          'Iteration: {}'.format(iteration),
                          'Loss: {}'.format(loss_value))

    def generate_document(self,predIter = None, selection = None, k = None, prob = None):
        """
            Generate documents.
            selection: The type of selection will be used. its either topk or nucleus, t and n can also be used.
            k: In case of topk, it represents the k value.
            prob: In case of nucleus, its represnts the probability thresholds 
            predIter: The number of iterations for the prediction loop unless the model generated a stop token.


            They are optional and the values used while creating the generator object will be used.
        """
        return super(GRUGenerator, self).generateDocument('GRU',predIter,selection,k,prob)


    


# gen = GRUGenerator(db, 2, 16, 64, 16, 2, 0, 'fasttext', 12, selectionParams={'sType': 't'})
# gen = GRUGenerator(db, 2, 16, 64, 16, 5, 0, 'fasttext', 12, selectionParams={'sType': 'n'})
# gen = GRUGenerator(db, 2, 16, 64, 16, 5, 0, 'fasttext', 12, selectionParams={'sType': 't', 'of': 10})


# gen = GRUGenerator(db, 2, 16, 50, 16, 5, 0, 'glove', 12,glovePath=r"C:\Users\Fahed\Desktop\PC\codes\glove.6B\glove.6B.50d.txt", selectionParams={'sType': 'n', 'probThreshold': 0.5})

# help(GRUGenerator)
# path = r"C:\Users\Fahed\Desktop\PC\codes\new_kw_clean3.txt"
# db = Preprocessing(path,seedParams={})

# gen = GRUGenerator(db, 1, 16, 64, 16, 2, 0, 'fasttext', 12, selectionParams={'sType': 't'})
# gen.run()

# for _ in range(5):
#     print(gen.generate_document())