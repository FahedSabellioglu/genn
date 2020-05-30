import numpy as np
from collections import Counter
import os
from torchtext.data import Field, Example, Dataset
import en_core_web_sm


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

    """


    def __init__(self,fileName,
                    spacyObj = None,
                    instanceMxLen = None,
                    fieldParams = {'lower': True}, 
                    seedParams = {'N_first': 1, 'minFreq': 5},
                     ):


        # To be set by the user
        self.fileName = self.__checkFileName(fileName)
        self.instanceMxLen = instanceMxLen
        self.seedParams = self.__checkSeedParams(seedParams)


        # class variables
        self.examples = None
        self.__nlp = self.__checkSpacyObj(spacyObj)
        self.__DataVocab = Field(fieldParams)
        self.__text = None

        self.__readFile()

    @staticmethod
    def classParams():
        print("************Params Info*************")
        print("FileName: The file where the data will be taken from, it has to be a text file. ")
        print("spacyObj: Spacy Language object to be used as entity recognition. For English: en_core_web_sm.load()")
        print("fieldParams: The Field obj parameters from the library torchtext.data")
        print("SeedParams: The parameters \n 1- N_first: create a seed from the first n_first words of the instance\n 2- minFreq: The min frequence need for a word to be considered as a seed. ")


    def info(self):
        print("************INFO*************")
        print("1- File name: ", self.fileName)
        print('2- randomSeed enabled: ', True if len(self.seedParams) else False )
        print("3- Unique tokens in vocabulary: ", len(self.__DataVocab.vocab))
        print("4- Number of pair instances: ", len(self.examples))
    
    @property
    def DataVocab(self):
        return self.__DataVocab

    def __checkFileName(self,fileName):
        if not fileName.split(".")[-1] == 'txt':
            raise Exception("Only works with txt files")
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
        combinde_tokens = [entity for entity in self.__nlp(instance).ents  if entity.label_ == "GPE" and 
                                       len(entity.text.split())>= 2]
        instance = instance.split()
        if combinde_tokens:
            start, end = combinde_tokens[0].start, combinde_tokens[0].end
            instance[start:end] = [' '.join(instance[start:end])]
            return instance
        
        return instance

    def __tokenize(self,instance):

        if self.__nlp:
            tokenized = self.__spacyTokenization(instance)
            return {'src': tokenized, 'trg': tokenized[1: ]}

        instance = instance.split()
        
        return {'src': instance, 'trg': instance[1: ]}

    def __lengthLimit(self,instance):
        if self.instanceMxLen:
            return len(instance.split()) <= self.instanceMxLen
        return True


    def __getObjects(self):        
        self.fields = {"src": self.__DataVocab}
        return [DocumentExample(**self.__tokenize(instance)) 
                for instance in self.__text
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
        self.__text = file_obj.read().split("\n")
        self.examples = self.__getObjects()
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
    