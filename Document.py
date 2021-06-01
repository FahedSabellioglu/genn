from torchtext.legacy.data import Example

class Document(Example):
    def __init__(self,src,trg):
        self.src = src
        self.trg = trg
        self.src_token = None
        self.trg_token = None
        self.src_length = len(self.src)
        self.trg_length = len(self.trg)
    
    def create_tokens(self,field):        
        self.src_token = [field.vocab.stoi[word] for word in self.src]
        self.trg_token = [field.vocab.stoi[word] for word in self.trg] + [field.vocab.stoi[field.eos_token]]

    def __str__(self):
        return '\n'.join([f'{attr}: {value}' for attr, value in vars(self).items()])