from genn.GPT2 import GPT2

class GPT2Summarizer(GPT2):
    def __init__(self, fileName,
                        epochs,
                        taskToken = "",
                        variant = "small",
                        batchSize = 32,
                        eos = "<|endoftext|>",
                        instanceMxLen = None,
                        txtSeparator = '\n',
                        csvIndex = None,
                        jsonKey = None,
                        seedParams = {'N_first': 1, 'minFreq': 5},
                        optimParams = {"lr" : 3e-4},
                        schedParams = {"warmup_steps" : 400}):
        super(GPT2Summarizer, self).__init__(
            fileName, taskToken, epochs, variant, batchSize, eos, instanceMxLen,
            txtSeparator, csvIndex, jsonKey, seedParams, optimParams, schedParams)

    def summarize_document(self,
                          n,
                          source=None,
                          isNucleus=True,
                          instanceMxLen=None,
                          k=None,
                          p=None,
                          noRepetition=False):
        self.model.eval()
        res = set()
        max_len = instanceMxLen if instanceMxLen!=None else self.instanceMxLen
        with torch.no_grad():
            while len(res) < n:
                cur_ids = torch.tensor(self.tokenizer.encode(source + " =")).unsqueeze(0).to(self.device)
                seed_len = cur_ids.shape[1]
                while cur_ids.shape[1] - seed_len < max_len:
                    outputs = self.model(cur_ids, labels=cur_ids)
                    _, logits = outputs[:2]

                    if isNucleus:
                        if p!=None:
                            next_token_id = self.select_nucleus(logits[0,-1], p=p)
                        next_token_id = self.select_nucleus(logits[0,-1])

                        
                    else: #topk
                        softmax_logits = torch.softmax(logits[0,-1], dim=0)
                        if k!=None:
                            next_token_id = self.choose_from_top(softmax_logits.to('cpu').numpy(), n=k)
                        next_token_id = self.choose_from_top(softmax_logits.to('cpu').numpy())

                    if noRepetition:
                        if self.isRedundant(cur_ids, next_token_id):
                            continue
                        
                    cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(self.device) * next_token_id], dim = 1)
                    if next_token_id in self.tokenizer.encode(self.eos):
                        break

                doc = self.tokenizer.decode(list(cur_ids.squeeze().to('cpu').numpy())).strip()
                res.add(doc)

        return [summ.split("=")[1] for summ in list(res)]