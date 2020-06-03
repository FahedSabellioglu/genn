import os

import numpy as np

from importers import *


class PretrainedEmbeddings:
    """
        Get pretrained embeddings using GloVe or fastText.
        
        Parameters
        ----------
        datasetObj : Preprocessing class object
            An object created using the class Preprocessing.
        
        embSize: int
            The size of the vector in which each will be embedded.

        glovePath: string, optional, default None
            The path for the glove embeddings if embeddingType is set to 'glove'.
        
        fastTextParams: dict, optional, default {}
            The parameters will be used while fine-tuning fastText to create the embeddings.
            Emtpy dictionary means the default training parameters will be used.
        
        loadFastText: string, optional, default None
            A path to the pretrained fastText model file.
        
        saveFastText: string, optional, default None
            A directory path to save the model with the name fastText_{dim}.bin
            or a file path that will have a filename with the extension .bin
    """

    def __init__(self,datasetObj, embSize = 64,
                                 embeddingType = 'fasttext',
                                 glovePath = None,
                                 fastTextParams = {},
                                 loadFastText = None,
                                 saveFastText = None,
                                  ):
        

        self.__preprocessingObj(datasetObj)
        self.__Datavocab = datasetObj.DataVocab.vocab.stoi
        self.__dataFile = datasetObj.fileName
        self.embSize = embSize


        self.__isGlove = self.__checkEmbedding(embeddingType)

        # fastText
        self.__loadFastText = self.__checkModelFile(loadFastText)
        self.__saveFastText = self.__saveFastTextParam(saveFastText)
        self.__fastTextParams = fastTextParams
        self.__fastTextModel = None

        # Glove
        self.__gloveEmb = {}
        self.glovePath = glovePath


        self.weights = self.__getEmbs()
    
    def __getEmbs(self):
        if self.__isGlove:
            self.__checkGloveParams()
            self.__getGlove()
            
        else:
            self.__getFast()

        return self.__getMatrix()

    def __checkEmbedding(self,value):
        value = value.lower()
        if value in 'glove': # both glove and g will be accepted.
            return True
        elif value in 'fasttext':
            self.__fasttext = importFasttext()
            return False

        raise Exception("Embedding type can only be glove or fasttext")

    def __preprocessingObj(self,preprobObj):
        if preprobObj.__class__.__name__ != 'Preprocessing':
            raise Exception("Please provide the dataset object of type Preprocessing.")
        return preprobObj

    def __checkModelFile(self,fileName):
        """
            ensures that the extension of fasttext trained file is .bin
        """
        if type(fileName) == str:
            if fileName.split(".")[-1] != 'bin':
                raise Exception("FastText trained model should have a .bin extension")
            return fileName
        return None


    def __embeddingType(self):
        return 'Glove' if self.__isGlove else 'FastText'

    def __getFast(self):
        
        if self.__loadFastText:
            self.__fastTextModel = self.__fasttext.load_model(self.__loadFastText)
            print("Loaded the model located at '{m}'".format(m= self.__loadFastText))

        else:
            self.__trainFast()

    def __saveFastTextParam(self, value):
        if value:
            if os.path.isdir(value):
                modelName = "fastText_{dim}.bin".format(dim=self.embSize)
                return os.path.join(value,modelName)
            elif value.endswith('.bin'):
                return value
            else:
                raise Exception("SaveFastText param is invalid, its neither a dirc path nor a .bin file path")
        return None

    def __trainFast(self):
        print("{pre_emb}: Training model".format(pre_emb=self.__embeddingType()))
        self.__fastTextModel = self.__fasttext.train_unsupervised(input=self.__dataFile,dim=self.embSize,**self.__fastTextParams)

        if self.__saveFastText:
            self.__fastTextModel.save_model(self.__saveFastText)
    

    def __checkGloveParams(self):
        """
            if isGlove is true, then glove path has to be provided
        """
        if self.__isGlove and not self.glovePath:
            raise Exception("Glove file path is needed when setting embeddingType=glove.")
    

    def __getGlove(self):
        with open(self.glovePath, 'r',encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in self.__Datavocab:
                    self.__gloveEmb[word] =  np.asarray(values[1:], "float32")
            
            if len(values[1:]) != self.embSize:
                self.embSize = len(values[1:])
                print("Setting embedding size using Glove file.")
                    
        print("{pre_emb}: READ {t} tokens".format(pre_emb=self.__embeddingType(),t = len(self.__gloveEmb.keys())))
    


    def __getMatrix(self):
        counter = 0
        weights_matrix = np.zeros((len(self.__Datavocab),self.embSize))    
        for index, word in zip(self.__Datavocab.values(),self.__Datavocab):

            if self.__isGlove:
                if word not in self.__gloveEmb:
                    weights_matrix[index] = np.random.normal(scale=0.6, size=(self.embSize, ))
                else:
                    weights_matrix[index] = self.__gloveEmb[word]
                    counter += 1
            else:
                weights_matrix[index] = self.__fastTextModel.get_word_vector(word)
                counter += 1
        
        weights_matrix[:2] = np.zeros(self.embSize,)
        print("{pre_emb}: FOUND {c} tokens out of {t}".format(pre_emb=self.__embeddingType(),c=counter,t=len(self.__Datavocab)))
        return weights_matrix
