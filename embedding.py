import numpy as np
from genn.importers import importFasttext


class PretrainedEmbeddings:
    
    def __init__(self, preprobObj, embSize = 64, embeddingOption = 'fasttext',
                     glovePath = None, fastTextParams = {}, loadFastText = None,
                     saveFastText = None):
        
        self.__Datavocab = preprobObj.getVocab.vocab.stoi
        self.__dataFile = preprobObj.getFileName
        self.__embSize = embSize

        self.__glovePath = glovePath
        self.__isGlove = self.__checkEmbedding(embeddingOption)

        self.__fastTextParams = fastTextParams
        self.__loadFastText = self.__checkModelFile(loadFastText)
        self.__saveFastText = saveFastText
        self.__gloveEmb = {}

        self.weights = self.__getEmbs()

    def __checkEmbedding(self,value):
        value = value.lower()
        if value in 'glove':
            if not self.__glovePath:
                raise Warning("GloVe file path is needed when setting embeddingOption=glove.")
            return True
        elif value in 'fasttext':
            self.__fasttext = importFasttext()
            print("Imported fasttext")
            return False

        raise Warning("Embedding type can only be glove or fasttext.")

    def __embeddingType(self):
        return 'GloVe' if self.__isGlove else "fastText"

    def __getEmbs(self):
        if self.__isGlove:
            self.__getGloVe()
        else:
            self.__getFast()

        return self.__getMatrix()


    def __getGloVe(self):
        with open(self.__glovePath, 'r',encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in self.__Datavocab:
                    self.__gloveEmb[word] =  np.asarray(values[1:], "float32")
            
            if len(values[1:]) != self.__embSize:
                self.__embSize = len(values[1:])
                print("Setting embedding size using the GloVe file.")
                    
        print("{pre_emb}: READ {t} tokens".format(pre_emb=self.__embeddingType(),t = len(self.__gloveEmb.keys())))


    def __getFast(self,):
        if self.__loadFastText:
            self.__fastTextModel = self.__fasttext.load_model(self.__loadFastText)
            print("Loaded the model located at '{m}'".format(m= self.__loadFastText))

        else:
            self.__trainFast()

    def __trainFast(self):
        print("{pre_emb}: Training model".format(pre_emb=self.__embeddingType()))
        self.__fastTextModel = self.__fasttext.train_unsupervised(input=self.__dataFile,dim=self.__embSize)

        if self.__saveFastText:
            self.__fastTextModel.save_model(self.__saveFastText)

    def __checkModelFile(self,fileName):
        """
            ensures that the extension of fasttext trained file is .bin
        """
        if type(fileName) == str:
            if fileName.split(".")[-1] != 'bin':
                raise Exception("FastText trained model should have a .bin extension")
            return fileName
        return None
    
    def __getMatrix(self):
        counter = 0
        weights_matrix = np.zeros((len(self.__Datavocab),self.__embSize))    
        for index, word in zip(self.__Datavocab.values(),self.__Datavocab):

            if self.__isGlove:
                if word not in self.__gloveEmb:
                    weights_matrix[index] = np.random.normal(scale=0.6, size=(self.__embSize, ))
                else:
                    weights_matrix[index] = self.__gloveEmb[word]
                    counter += 1
            else:
                weights_matrix[index] = self.__fastTextModel.get_word_vector(word)
                counter += 1
        
        print("{pre_emb}: FOUND {c} tokens out of {t}".format(pre_emb=self.__embeddingType(),c=counter,t=len(self.__Datavocab)))
        return weights_matrix