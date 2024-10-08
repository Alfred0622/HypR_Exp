def getBertPretrainName(dataset):
    if (dataset in ['aishell', 'aishell2', 'AISHELL1']):
        return 'bert-base-chinese'
    elif (dataset in ['tedlium2', 'librispeech', 'TEDLIUM2', 'LibriSpeech']):
        return 'bert-base-uncased'
    elif (dataset in ['csj', 'CSJ']):
        return 'cl-tohoku/bert-base-japanese'

def getGPTPretrainName(dataset):
    if (dataset in ['aishell', 'aishell2', 'AISHELL1', 'AISHELL2']):
        return 'bert-base-chinese'
    elif (dataset in ['tedlium2', 'librispeech', 'TEDLIUM2', 'LibriSpeech']):
        return 'gpt2'
    elif (dataset in ['csj', 'CSJ']):
        return 'ClassCat/gpt2-base-japanese-v2'

def getBartPretrainName(dataset):
    if (dataset in ['AISHELL1', 'AISHELL2']):
        return 'fnlp/bart-base-chinese'
    elif (dataset in ['TEDLIUM2', 'LibriSpeech']):
        return 'facebook/bart-base'
    elif (dataset in ['CSJ']):
        return 'ku-nlp/bart-base-japanese'