import numpy as np
import _pickle as pickle
import os.path as op
import random
import glob
import numpy as np
import time
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator
import torch
from collections import Counter
from tqdm import tqdm
import math
torch.manual_seed(314)
SAVE_PARAMS_EVERY = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tic = time.process_time()
import warnings
warnings.filterwarnings("ignore")
class Word2vec:
    def __init__(self) -> None:
        self.window_size=10
        self.dim_vectors=100
        self.tablesize = 1000000
        self.counter = Counter()

    def load_data_sentences(self):
        first = True 
        with open('malayalam_senteces2.txt','r',encoding='utf-8') as file:
            for line in file:
                line = re.sub(r'[^\u0D00-\u0D7F\s]', ' ', line).split()
                self.counter.update(line)
                yield line
        

    def tokens(self):
        self.vocab=build_vocab_from_iterator(self.load_data_sentences()).to(device)
        self.vocab.freqs=self.counter


    def get_train_data(self):
        self.train_data=[]
        self.train_labels=[]
        for line in self.load_data_sentences():
            for i in range(len(line)):
                input_word=self.vocab.lookup_indices([line[i]])[0]
                #input_word=self.tokens_id[line[i]]
                target_words=[]
                for j in range(max(0, i - self.window_size), i):
                    target_words.append(self.vocab.lookup_indices([line[j]])[0])
                for j in range(i + 1, min(len(line), i + self.window_size + 1)):
                    target_words.append(self.vocab.lookup_indices([line[j]])[0])
                if len(target_words)!=0:
                    self.train_data.append(input_word)
                    self.train_labels.append(target_words)
        self.train_data=torch.tensor(self.train_data).to(device)
        self.train_labels=[torch.tensor(label).to(device) for label in self.train_labels]
     
    def dataloader(self,batch_size=1000):
        for i in range(0,len(self.train_data),batch_size):
            yield self.train_data[i:i+batch_size], self.train_labels[i:i+batch_size]

    def sampleTable(self):
        #if hasattr(self, '_sampleTable') and self._sampleTable is not None:
        #    return self._sampleTable
        nTokens = self.vocab.__len__()
        #print(nTokens)
        samplingFreq = torch.zeros((nTokens,))
        i = 0
        for w in range(nTokens):
            w = self.vocab.lookup_token(w)
            #print(w)
            if w in self.vocab.freqs:
                freq = 1.0 * self.vocab.freqs[w]               
                freq = freq ** 1
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1
        samplingFreq /= torch.sum(samplingFreq)
        self._sampleTable = torch.multinomial(samplingFreq, self.tablesize, replacement=True)
        #return self._sampleTable


    def sampleTokenIdx(self):
        #print("taking sample token")
        return self._sampleTable[torch.randint(0, self.tablesize - 1,(1,))]

    
    def load_saved_params(self):
        st = 0
        for f in glob.glob("saved_params_*.pth"):
            iter = int(op.splitext(op.basename(f))[0].split("_")[2])
            if (iter > st):
                st = iter
        if st > 0:
            checkpoint = torch.load("saved_params_%d.pth" % st)
            self.torch_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            torch.set_rng_state(checkpoint['state'])
            return st
        else:
            return st



    

    def save_params(self, iter, params):
        
            checkpoint = {
        'iter': iter,
        'model_state_dict': self.torch_model.state_dict(),  # Save the model weights
        'optimizer_state_dict': self.optimizer.state_dict(),  # Save the optimizer state
        'state':torch.get_rng_state()
    }
            torch.save(checkpoint, "saved_params_%d.pth" % iter)
       




    def model_torch(self,):
        if not hasattr(self,'torch_model'):
            self.torch_model=nn.Sequential(nn.Embedding(self.vocab.__len__(),self.dim_vectors),nn.Linear(self.dim_vectors,self.vocab.__len__()))
            self.optimizer = torch.optim.SGD(self.torch_model.parameters(), lr=self.lr)
            self.loss=nn.BCEWithLogitsLoss()



    
    
    def negative_sampling(self,input_index, target_index, K=10,skip_gram=True):
        #print("started neg sample")
        indices = [target_index]
        for k in range(K):
            newidx = self.sampleTokenIdx()
            while newidx == target_index:
                newidx = self.sampleTokenIdx()
            indices += [newidx]
        directions = torch.tensor([1] + [-1 for _ in range(K)], dtype=torch.float32)

        
        self.torch_model.train()
        self.optimizer.zero_grad()
        input_vector=self.torch_model[0](input_index)
        output_vector=self.torch_model[1].weight[torch.tensor(indices).to(device)]
        logits = torch.matmul(output_vector, input_vector.view(-1, 1)).squeeze() * directions  # Shape adjustment
        labels = torch.tensor([1] + [0] * K, dtype=torch.float32) 
        loss= self.loss(logits,labels)
        loss.backward()
        self.optimizer.step()
        #print("finished neg sampling")
        return loss.item()



        

    def skipgram(self,current_word, context_words, tokens, word2vec_cost_grads):
        cost = 0.0
        #print("started word")
        for context in context_words:
            #try:
                #print(self.vocab[idx],self.vocab[tokens[context]])
                cost = word2vec_cost_grads(current_word,context,True)
                #print(cost)
                
            #except:
                #print(idx,tokens[context])
        #print("finished word")
        return cost
    
    def word2vec_sgd_wrapper(self,word2vec_model, tokens,C, word2vec_gradient):
        batchsize = 1000
        cost = 0.0
        for train_data,train_label in tqdm(obj.dataloader(batchsize),total=math.ceil(len(self.train_data)/batchsize),desc='Batch Progress',unit='batch'):
            for centerword, context in zip(train_data,train_label):
           
                c = word2vec_model(centerword, context, tokens,
                                            word2vec_gradient)
            cost += c / batchsize

        return cost

    
    def sgd(self,f, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=100):
        self.lr=step
        self.model_torch()
        self.sampleTable()
        ANNEAL_EVERY = 20000
        if useSaved:
            start_iter= self.load_saved_params()
            if start_iter > 0:
                step *= 0.5 ** (start_iter / ANNEAL_EVERY)
                self.lr=step
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            
        else:
            start_iter = 0
       
        if not postprocessing:
            postprocessing = lambda x: x
        expcost = None
        x=self.torch_model.state_dict()
        for iter in range(start_iter + 1, iterations + 1):
            print(f'Epoch {iter}:')
            cost= f()
            print("Cost: ",cost)
            x=self.torch_model.state_dict()
            if iter % PRINT_EVERY == 0:
                print("iter %d: %f" % (iter, cost))
            if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
                self.save_params(iter, x)
            if iter % ANNEAL_EVERY == 0:
                step *= 0.5
        
        return x
    

    def initialize(self):
        
        self.n_words=len(self.vocab)
        
        #word_vectors = np.concatenate(((np.random.rand(self.n_words, self.dim_vectors) - .5) /  self.dim_vectors, np.zeros((self.n_words, self.dim_vectors))), axis=0)
        word_vectors0 = self.sgd(lambda : self.word2vec_sgd_wrapper(self.skipgram, self.vocab.__len__(),  self.window_size, self.negative_sampling),
                             0.01, 1, None, True, PRINT_EVERY=100)
        word_vectors0=word_vectors0['0.weight']#+word_vectors0['1.weight']
        return word_vectors0
    

obj=Word2vec()
obj.tokens()
print(obj.vocab.__len__())
obj.get_train_data()
wordvectors=obj.initialize()
print("No of Tokens: ",obj.n_words)
toc = time.process_time()
print("Training time: " + str(toc-tic))
print("Vector shape: ",wordvectors.shape)
visualizeWords = ["life", "death", "promising", 
	"good", "great", "cool",  "wonderful", "well", "amazing",
	"worth", "sweet", "enjoyable", "bad", "waste", 
	"annoying"]

visualizeIdx = obj.vocab.lookup_indices(visualizeWords)
#wordvectors=wordvectors[0]
print(wordvectors)
visualizeVecs = wordvectors[visualizeIdx].tolist()
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
import matplotlib.pyplot as plt

'''
coord = temp.dot(U[:,0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
    	bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors.png')
plt.show()
'''
coord = temp.dot(U[:,0:3])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(visualizeWords)):
    ax.text(coord[i, 0], coord[i, 1], coord[i, 2], visualizeWords[i],
            bbox=dict(facecolor='green', alpha=0.1))

# Set the limits for x, y, and z axes
ax.set_xlim([np.min(coord[:, 0]), np.max(coord[:, 0])])
ax.set_ylim([np.min(coord[:, 1]), np.max(coord[:, 1])])
ax.set_zlim([np.min(coord[:, 2]), np.max(coord[:, 2])])



plt.savefig('word_vectors.png')
plt.show()


