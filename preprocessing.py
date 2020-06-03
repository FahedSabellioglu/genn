import csv
import json
import os
import re
from collections import Counter

import numpy as np
from torchtext.data import Dataset, Example, Field

from importers import *


class DocumentExample(Example):
    def __init__(self,src,trg):
        self.src = src
        self.trg = trg
        self.src_token = None
        self.trg_token = None
        self.src_length = len(self.src)
        self.trg_length = len(self.trg)
    
    def create_tokens(self,field):        
        self.src_token = [field.vocab.stoi[word] for word in self.src]
        self.trg_token = [field.vocab.stoi[word] for word in self.trg] + [0]  


class Preprocessing(Dataset):
    """
        Preprocessing the documents, creating paris of src and trg and building the vocab.

        Parameters
        ----------
        fileName: string
            The path to a text file with training documents/instances separated by newline (\n).

        spacyObj: spaCy language class object, optional, default None
            The spacy language class object that will be used in recognizing entities
            and grouping them into one token. 

        instanceMxLen: int, optional , default None
            Used to ignore documents that are larger than a certain length of tokens.
        
        fieldParams: dict, optional, default {'lower': True}
            The parameters that will be used in creating the Field object.
        
        seedParams: dict, optional, default {'N_first': 1, 'minFreq': 5}
            Parameters that will be used while selecting the random seed used to initalize prediction.
                N_first: get the first n_first tokens of each training document.
                minFreq: The min frequency a seed must have before it is included in the pool. 
            
            by default random seed is enabled. If seedParams = {} then static seed will be applied.

        txtSeparator: str, optional, default '\n'   
            When the data file is a txt file, this is the character serparating documents.
            By default, the assumption is that each document is on a separate line.

        csvIndex: int, optional, default None
            When the data file is a csv file, this is the index of the column to parse from.

        jsonKey: str, optional, default None
            When the data file is a csv file, this is the json key to parse from.
            Typically, it is 'body' or 'text'.

    """

    __tokPattern = r"""[0-9A-Za-z_]*[A-Za-z_-]+[0-9A-Za-z_]*|\.|\!|\?|\d+|\-|%|[.,!?;'"]"""

    def __init__(self, fileName,
                    tokenizationMethod = 'regex',
                    spacyObj = None,
                    instanceMxLen = None,
                    fieldParams = {'lower': True, 'eos_token': '<!EOS!>'}, 
                    seedParams = {'N_first': 1, 'minFreq': 5},
                    txtSeparator = '\n',
                    csvIndex = None,
                    jsonKey = None):


        # To be set by the user
        self.csvIndex = csvIndex
        self.jsonKey = jsonKey
        self.txtSeparator = txtSeparator

        self.fileName = self.__checkFileName(fileName)
        self.instanceMxLen = instanceMxLen
        self.seedParams = self.__checkSeedParams(seedParams)


        # class variables
        self.fieldParams = fieldParams
        self.examples = None
        self.__nlp = self.__checkSpacyObj(spacyObj)
        self.__DataVocab = Field(**self.fieldParams)
        self.__text = None


        self.__customTokenize = self.__tokenizationMethod(tokenizationMethod)

        self.__readFile()


    def info(self):
        print("************INFO*************")
        if self.extension == "json":
            print(f"File name: {self.fileName}, jsonKey = {self.jsonKey}")
        elif self.extension == "txt":
            print(f"File name: {self.fileName}, txtSeparator = {repr(self.txtSeparator)}")
        else:
            print(f"File name: {self.fileName}, csvIndex = {self.csvIndex}")
        print("Tokenization method:", self.__customTokenize.__name__)
        print("Ignore long documents:", f"True, where length > {self.instanceMxLen}" if self.instanceMxLen!= None 
                                                    else False)
        print("Seed parameters:", self.seedParams)
        print("Field parameters:", self.fieldParams)

    
    @property
    def DataVocab(self):
        return self.__DataVocab

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
    
    def __checkSpacyObj(self,spacyObj):

        if spacyObj:
            if type(spacyObj).__base__.__name__ != 'Language':
                raise Exception("spacyObj has to be of a type Language")

            return spacyObj

        return None

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, id):
        return self.examples[id]

    def __spacyTokenization(self,instance):

        return [entity.text.strip() for entity in self.__nlp(instance) if entity.text.strip() ]


    def nltkTokenization(self, document):
        return self.word_tokenize(document)

    def regexTokenization(self, document):
        return re.findall(self.__tokPattern, document)

    def __tokenizationMethod(self,value):
        value = value.lower()
        if value in 'nltk':
            self.word_tokenize = importNltk()
            return self.nltkTokenization
        elif value in 'regex':
            return self.regexTokenization
        elif value in 'spacy':
            if not self.__nlp:
                raise Exception("The spacy object is not provided to tokenize using spacy.")
            return self.__spacyTokenization
        
        raise Exception("The param 'tokenizationMethod' can only be nltk, regex and spacy")

    def __tokenize(self,instance):

        instance = self.__customTokenize(instance)

        return {'src': instance, 'trg': instance[1: ]}

    def __lengthLimit(self, instance):
        length = len(instance.split()) 
        if self.instanceMxLen:
            return length <= self.instanceMxLen and length > 0
        return length > 0


    def __getObjects(self, rawDocuments):        
        self.fields = {"src": self.__DataVocab}
        return [DocumentExample(**self.__tokenize(instance)) 
                for instance in rawDocuments
                if self.__lengthLimit(instance)]

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

        self.examples = self.__getObjects(self.__text)
        self.seeds =  self.__getStartWords()
        self.__build_vocab()

    def __prepare_tokens(self):
        for instance in self.examples:
            instance.create_tokens(self.__DataVocab) 

    def __build_vocab(self):
        self.__DataVocab.build_vocab(self)
        self.__DataVocab.vocab.stoi['<unk>'] = 1
        self.__DataVocab.vocab.stoi['<pad>'] = 0
        self.__DataVocab.vocab.itos[0] = '<pad>'
        self.__DataVocab.vocab.itos[1] = '<unk>'
        self.__prepare_tokens() 

    def getSeed(self):
        """
            return a weighted seed. 
            In case static seed is enabled, then the most frequent token will be the seed.
        """
        seeds = list(self.seeds.keys())
        probs = list(self.seeds.values())
        return np.random.choice(seeds, 1 , probs).tolist()
