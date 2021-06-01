import csv, json, re 
from torchtext.legacy.data import Dataset, Field
from genn.importers import importNltk
from genn.controllers import checkSeedParams, checkFileParams, readFiles, getStartWords, checkParams
from genn.Document import Document
from collections import Counter
from numpy.random import choice



class Preprocessing(Dataset):
    
    __tokPattern = r"""[0-9A-Za-z_]*[A-Za-z_-]+[0-9A-Za-z_]*|\.|\!|\?|\d+|\-|%|[.,!?;'"]"""
    __supportedExtensions = ['txt', 'csv', 'json']
    __seedAttrs = ['nFirst', 'minFreq']


    def __init__(self,
                    fileParams = {}, tokenizationOption='regex',
                    seedParams = {'nFirst': 1, 'minFreq': 5},
                    fieldParams = {'lower': True, 'eos_token': '<!EOS!>'},spacyObj = None):

        self.__fileName, self.__fileExtension, self.__parsingColumn = checkFileParams(fileParams)
        self.__seedParams = checkSeedParams(seedParams)
        self.__DataVocab = Field(**fieldParams)
        self.__spacyObj = spacyObj
        self.__customTokenize = self.__tokenizationMethod(tokenizationOption)
        self.__readFile()
    
    @property
    def getFileName(self):
        return self.__fileName
    
    @property
    def getVocab(self):
        return self.__DataVocab

    def __readFile(self):
        text = readFiles(self.__fileName, self.__fileExtension, self.__parsingColumn)                
        self.examples = self.__getObjects(text)
        self.__seeds = getStartWords(self.__seedParams, text)
        self.__build_vocab()
    
    def __getObjects(self, text):   
        self.fields = {"src": self.__DataVocab}     
        return [Document(**self.__tokenize(instance)) 
                for instance in text]

    def __build_vocab(self):
        self.__DataVocab.build_vocab(self)
        for instance in self.examples:
            instance.create_tokens(self.__DataVocab) 

    def __regexTokenization(self, document):
        return re.findall(self.__tokPattern, document)

    def __nltkTokenization(self, document):
        return self.tokenizer(document)

    def __spacyTokenization(self,instance):
        return [entity.text.strip() for entity in self.__spacyObj(instance) if entity.text.strip() ]

    def __tokenize(self,instance):
        instance = self.__customTokenize(instance)
        return {'src': instance, 'trg': instance[1: ]}
    
    @checkParams(str)
    def __tokenizationMethod(self, param):
        param = param.lower()

        if param == 'nltk':
            self.tokenizer = importNltk()
            return self.__nltkTokenization

        elif param == 'regex':
            return self.__regexTokenization
        
        elif param == 'spacy':
            if not self.__spacyObj:
                raise Exception("Please provide the spacy object to tokenize with.")
                
            return self.__spacyTokenization
                
        
        raise Exception("The parameter 'tokenizationOption' can only be nltk, regex and spacy")

    def getSeed(self):
        """
            return a weighted seed. 
            In case static seed is enabled, then the most frequent token will be the seed.
        """
        seeds = list(self.__seeds.keys())
        probs = list(self.__seeds.values())
        return choice(seeds, 1 , probs).tolist()


            
            





