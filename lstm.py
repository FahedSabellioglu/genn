import time

import torch
import torch.nn as nn

from generator import Generator
from preprocessing import Preprocessing


class LSTMGenerator(Generator):
    """
        LSTM model for text generation.
        Parameters
        ----------
        datasetObj : Preprocessing class object
            An object created using the class Preprocessing.

        nLayers: int
            The number of LSTM layers.
        
        batchSize: int
            The number of training instances per iteration.

        embSize: int
            The size of the vector in which each token will be embedded.
        
        lstmSize: int
            The number of features in the hidden state.
        
        epochs: int
            The number of cycles over the training data.
        
        dropout: int, optional, default 0
            The percentage of units that will be dropped per layer.
        
        embeddingType: string, optional, default fasttext
            The type of embeddings that will be used for the tokens.
            It can be glove, fasttext or default. Default means the NN.Embedding will be used.
        
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
            sType: The type of selection method to use during prediciton. It is either 'topk' or 'nucleus', 't' and 'n' can also be used.
            k: in case of topk, it represents the k value.
            probThreshold: When using Nucleus selection, this represnts the probability threshold p. 

    """
    def __init__(self,datasetObj, nLayers, batchSize, embSize, lstmSize,
                   epochs, dropout = 0, embeddingType ='fasttext',
                   predictionIteration = 10, glovePath = None, fastTextParams = {},
                   loadFastText = None, saveFastText = None, fineTuneEmbs = False, 
                   selectionParams = {"sType": 'topk', 'k': 5}):

        super(LSTMGenerator, self).__init__(datasetObj,nLayers,batchSize, embSize,
                                           lstmSize,epochs,dropout,embeddingType,predictionIteration,
                                           glovePath,fastTextParams,loadFastText,saveFastText,fineTuneEmbs,
                                           selectionParams)

        self.lstm_size = self.rnnSize
        self.predIter = predictionIteration

        self.lstm = nn.LSTM(self.embSize,
                        self.lstm_size,
                        num_layers = self.nLayers,
                        dropout = self.dropout,
                        batch_first = True)
        self.dense = nn.Linear(self.lstm_size, self.n_vocab)


    def forward(self, x, length, prev_state):
        """
            params:
                x: the tensor of input instances.
                length: the length that will be used from each training instance to handle paddings.
                prev_state: the hidden and the cell states

        """

        embed = self.embedding(x)
        packed_input = self.pack_src(embed, length)
        packed_output, state = self.lstm(packed_input, prev_state)
        padded,_ = self.pad_pack(packed_output) 
        logits = self.dense(padded)
        return logits, state

    def info(self):
        """
            Show the parameters used with the generator.
        """

        super(LSTMGenerator, self).info('LSTM')
    
    def run(self):
        """
            Training method
        """

        criterion, optimizer = self.get_loss_and_train_op(0.01)

        iteration = 0
        total_iterations = (len(self.db) // self.batchSize) * self.epochs

        for e in range(self.epochs):        
            batches = self.get_batches()
            
            state_h, state_c = self.zero_state(2, self.batchSize)
            state_h = state_h.to(self.device)
            state_c = state_c.to(self.device)
            
            batch_times = []
            for batch in batches:

                start_time = time.time()
                src_lengths, _ ,src_batch,trg_batch = self.get_src_trg(batch)
                
                iteration += 1

                self.train()


                optimizer.zero_grad()
                
                trg_batch = torch.t(trg_batch).to(self.device)
                src_batch = src_batch.to(self.device)
                
                logits, (state_h, state_c) = self(src_batch,src_lengths,(state_h,state_c))

                loss = criterion(logits.transpose(1, 2), trg_batch)

                state_h = state_h.detach()
                state_c = state_c.detach()

                loss_value = loss.item()

                loss.backward(retain_graph=True)

                optimizer.step()

                loss.backward()

                _ = torch.nn.utils.clip_grad_norm_(
                self.parameters(), 5)

                optimizer.step()

                batch_time = time.time() - start_time
                batch_times.append(batch_time)

                if iteration % 10 == 0:
                    mean_time = sum(batch_times)/len(batch_times)
                    remaining_iters = total_iterations - iteration       
                    remaining_seconds = remaining_iters * mean_time 
                    remaining_time = time.strftime("%H:%M:%S",
                        time.gmtime(remaining_seconds))
                    progress = "{:.2%}".format(iteration/total_iterations)
                    print('Epoch: {}/{}'.format(e+1, self.epochs),
                          'Progress:', progress,
                          'Loss: {}'.format(loss_value),
                          'ETA:', remaining_time)


    def generate_document(self, predIter = None, selection = None, k = None, prob = None):
        """
            Generate documents.
            selection: The type of selection will be used. its either topk or nucleus, t and n can also be used.
            k: In case of topk, it represents the k value.
            prob: In case of nucleus, its represnts the probability thresholds 
            predIter: The number of iterations for the prediction loop unless the model generated a stop token.


            They are optional while calling this method. The values passed while creating the generator object will be used.
        """
        return super(LSTMGenerator, self).generateDocument('LSTM', predIter, selection, k, prob)

