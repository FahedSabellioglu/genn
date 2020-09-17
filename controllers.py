import numpy as np
import torch
import torch.nn.functional as F
import json, csv, time
from collections import Counter

def assertType(provided, expected, name):
    assert isinstance(provided,expected), f"{name} expected type {expected}, but instead got {type(provided)}"

def checkParams(*decParam):
    def warpFunc(calledFunc):
        paramName = calledFunc.__code__.co_varnames            
        index = 1 if 'self' in paramName else 0
        paramTypes = dict( zip(paramName[index: ], decParam )) 
        
        def controlParams(*args,**kwargs):
            provided = {}

            if args[index: ]:
                provided.update(dict(zip(paramName[index: ], args[index: ])))
            
            if kwargs:
                provided.update(kwargs)

            for param in provided:
                assertType(provided[param], paramTypes[param], param)

            return calledFunc(*args, **kwargs)
        return controlParams
    return warpFunc

def checkProb(value):
    if sum([type(value) == float, 0 <= value <= 1]) != 2:
        raise Exception("Make sure the probThreshold is a positive float, between 0 and 1." )
    return value


@checkParams(dict)
def checkFileParams(fileParams):
    supportedExtensions = ['txt', 'csv', 'json']

    if 'fileName' not in fileParams:
        raise Warning("fileParams has to be a dictionary with fileName for the data source\n"
                      "and txtSeperator key for txt files(default is '\\n') or a column key for csv and json files.")

    fileName = fileParams['fileName']
    fileExtension = fileName.split(".")[-1]

    if fileExtension in supportedExtensions:
        
        if fileExtension != 'txt' and 'column' not in fileParams:
            raise Warning("Please provide a 'column' key in fileParam dict to indicate the column/key to parse the data from.")

        elif fileExtension == 'txt':
            txtSeperator = fileParams.get('txtSeperator') or '\n'
            return fileName, fileExtension, txtSeperator
        
        return fileName, fileExtension, fileParams['column']

    raise Warning("Only works with txt, csv, and json files.")

@checkParams(dict)
def checkSeedParams(params):
    attr_list = ['nFirst', 'minFreq']

    if not len(params):
        return {}
    
    elif not all(attr in params for attr in attr_list):
        raise Exception("seedParams should contain the attributes {attrs}".format(attrs = attr_list))

    return params

def checkDropout(dropout, nLayers):
    if nLayers < 2 and dropout != 0:
        raise Exception("The network must have more than 1 layer to have dropout.")
    if not 0.0 <= dropout <= 1.0:
        raise Exception("Dropout value must be between 0 and 1.")     
    return dropout   

def chooseFromTop(probs, n=5):
    probs = torch.softmax(probs, dim= -1)
    tokensProb, topIx = torch.topk(probs, k=n)
    tokensProb = tokensProb / torch.sum(tokensProb) # Normalize
    tokensProb = tokensProb.cpu().detach().numpy()

    choice = np.random.choice(n, 1, p = tokensProb)
    tokenId = topIx[choice][0]
    return int(tokenId)


def selectNucleus(out, p = None):
    probs = F.softmax(out, dim=-1)    
    idxs = torch.argsort(probs, descending = True)
    res,prob, cumsum = [], [], 0.
    for idx in idxs:
        res.append(idx)
        prob.append(probs[idx].item())
        cumsum+=probs[idx]
        if cumsum > p:
            break
    prob = prob / np.sum(prob)
    choice = np.random.choice(res, 1, p = prob)[0]
    return choice

def parseData(fileObj, parsingColumn):
    return [row[parsingColumn] for row in fileObj]

def readFiles(filepath, fileExtension, parsingColumn):
    file_obj = open(filepath)

    if fileExtension == "txt":
        return file_obj.read().split(parsingColumn)

    elif fileExtension == "csv":
        reader = csv.DictReader(file_obj)
        return parseData(reader, parsingColumn)

    reader = json.loads(file_obj.read())
    return parseData(reader, parsingColumn)



def getStartWords(seedParams, text):
    nFirst = 1 if 'nFirst' not in seedParams else seedParams['nFirst']

    startWordsFreq = Counter([' '.join(instance.split()[:nFirst])
                        for instance in text
                        if len(instance.split()) > nFirst])
    
    if not len(seedParams):
        word, _ = startWordsFreq.most_common(1)[0]
        print("The word {w} is the Static Seed. ".format(w=word))
        return {word: 1.0}
    
    return {word:freq/len(startWordsFreq) for word,freq  in startWordsFreq.items() if freq >= seedParams['minFreq']}



def timeChecker(batchTimes, batchCount, batchesLen, curEpoch, totEpoch, lastLoss):
    meanTime = sum(batchTimes)/len(batchTimes)
    remainingBatches = batchesLen - batchCount       
    remainingSeconds = remainingBatches * meanTime 
    remainingTime = time.strftime("%H:%M:%S",
        time.gmtime(remainingSeconds))
    progress = "{:.2%}".format(batchCount/batchesLen)
    print('Epoch: {}/{}'.format(curEpoch+1, totEpoch),
            'Progress:', progress,
            'Loss: {}'.format(lastLoss),
            'ETA:', remainingTime)

