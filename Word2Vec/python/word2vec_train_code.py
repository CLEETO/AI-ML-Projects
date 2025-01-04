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
tic = time.process_time()
#data_to_save=[]
random.seed(314, version=1)
SAVE_PARAMS_EVERY = 5000

class Word2vec:
    def __init__(self) -> None:
        self.window_size=5
        self.dim_vectors=10
        self.tablesize = 1000000



   
# Replace characters in the text
        

    def load_data_sentences(self):
        self.sentences=[]
        first=True
        with open('dataset_sentences.txt','r',encoding='utf-8') as file:
            for line in file:
                if first:
                    first=False
                    continue
                line=line.strip().lower().replace('-lrb-', '(').replace('-rrb-', ')')
                
                #print(line)
                #line = re.sub(r'[^A-Za-z0-9\s]', '', line).split()  
                line=line.split()
                line=line[1:]
                #print(line)
                self.sentences += [[w.lower() for w in line]]
        

    def tokens(self):
        self.tokens_id={}
        self.token_freq={}
        self.vocab=[]
        idx=0
        self.wordcount=0
        for line in self.sentences:
            for word in line:
                self.wordcount+=1
                if word not in self.vocab:
                    self.tokens_id[word]=idx
                    self.token_freq[word]=1
                    self.vocab.append(word)
                    idx+=1
                else:
                    self.token_freq[word]+=1
        #print(self.token_freq)
        
        
                
    
    def one_hot_encoder(self,words):
        one_hot_word=[0]*len(self.vocab)
        for word in words:
            one_hot_word[self.tokens_id[word]]=1
        return one_hot_word
    


    def get_train_data(self):
        self.train_data=[]
        self.train_labels=[]
        for line in self.sentences:
            for i in range(len(line)):
                input_word=self.one_hot_encoder([line[i]])
                target_words=[]
                for j in range(max(0, i - self.window_size), i):
                    target_words.append(line[j])
                for j in range(i + 1, min(len(line), i + self.window_size + 1)):
                    target_words.append(line[j])
                target_words=self.one_hot_encoder(target_words)
                self.train_data.append(input_word)
                self.train_labels.append(target_words)

    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable
        nTokens = len(self.tokens_id)
        samplingFreq = np.zeros((nTokens,))
        self.allSentences()
        i = 0
        for w in range(nTokens):
            w = self.vocab[i]
            if w in self.token_freq:
                freq = 1.0 * self.token_freq[w]               
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1
        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize
        self._sampleTable = [0] * self.tablesize
        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j
        return self._sampleTable

    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]


    def rejectProb(self):
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb
        threshold = 1e-5 * self.wordcount
        nTokens = len(self.tokens_id)
        rejectProb = np.zeros((nTokens,))
        ids=self.tokens_id.values()
        for i in range(nTokens):
            w = self.vocab[i]    
            freq = 1.0 * self.token_freq[w]
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))
        self._rejectProb = rejectProb
        return self._rejectProb

    
    def load_saved_params(self):
        st = 0
        for f in glob.glob("params/numpy/saved_params_*.npy"):
            iter = int(op.splitext(op.basename(f))[0].split("_")[2])
            if (iter > st):
                st = iter
        if st > 0:
            with open("params/numpy/saved_params_%d.npy" % st, "rb") as f:
                params = pickle.load(f)
                state = pickle.load(f)
            return st, params, state
        else:
            return st, None, None



    def allSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences
        sentences = self.sentences
        rejectProb = self.rejectProb()
        tokens = self.tokens_id
        allsentences=[]       
        for sentece in sentences*30:         
            words=[]          
            for word in sentece:
                k=self.vocab.index(word)
                if 0>=rejectProb[k] or random.random() >= rejectProb[k]:
                    words+=[word]         
            if len(words)>1:
                allsentences+=[words]
        self._allsentences = allsentences
        return self._allsentences
    

    def save_params(self, iter, params):
        with open("params/numpy/saved_params_%d.npy" % iter, "wb") as f:
            pickle.dump(params, f)
            pickle.dump(random.getstate(), f)

    def sigmoid(self,Z):
        A = 1. / (1. + np.exp(-Z))
        cache = Z
        return A, cache
    
    def softmax(self,Z):
        if len(Z.shape) > 1:
            max_matrix = np.max(Z, axis=0)
            stable_Z = Z - max_matrix
            e = np.exp(stable_Z)
            A = e / np.sum(e, axis=0, keepdims=True)
        # Softmax implementation for vector.
        else:
            vector_max_value = np.max(Z)
            A = (np.exp(Z - vector_max_value)) / sum(np.exp(Z - vector_max_value))

        assert A.shape == Z.shape

        cache = Z

        return A, cache
    

    def relu(self,Z):
   
        A = np.maximum(0,Z)
        assert(A.shape == Z.shape)
        cache = Z
        return A, cache
        
    def linear_forward(self,A, W, b):
        Z = np.dot(W, A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache



    def linear_activation_forward(self,A_prev, W, b, activation):
    

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)

        if activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)

        if activation == "softmax":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.softmax(Z)

        if activation == "linear":
            # A particular case in which there is no activation function (useful for Word2Vec)
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = Z
            activation_cache = Z

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    
    def linear_backward(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db


    def softmax_cost_grads(self,input_vector, output_vectors, target_index, dataset):
        A_prev = input_vector.T
        W = output_vectors
        b = np.zeros((W.shape[0],1))
        activation = "softmax"
        probabilities, cache = self.linear_activation_forward(A_prev, W, b, activation)

        # Cost value
        cost = -np.log(probabilities[target_index])

        # Backward propagation

        # First step
        probabilities[target_index] -= 1 # (n_words, 1)
        delta_out = probabilities
        delta_out = delta_out.reshape(probabilities.shape[0]) # (n_words,)

        # Second step (gradients of weights of the second layer)
        grad_pred = np.dot(delta_out, output_vectors) # (1, dim_embed)

        # Third step (gradients of weights of the first layer)
        # See the implementation of the linear_backward(dZ, cache) in dnn.py
        _, grad, _ = self.linear_backward(delta_out.reshape(delta_out.shape[0], 1), cache[0]) # (n_words, dim_embed)

        return cost, grad_pred, grad



   

    def negative_sampling(self,input_vector, output_vectors, target_index, K=10):
        grad_pred = np.zeros_like(input_vector)
        grad = np.zeros_like(output_vectors)
        indices = [target_index]
        for k in range(K):
            newidx = self.sampleTokenIdx()
            while newidx == target_index:
                newidx = self.sampleTokenIdx()
            indices += [newidx]
        directions = np.array([1] + [-1 for k in range(K)])
        N = np.shape(output_vectors)[1]
        output_words = output_vectors[indices,:]
        input_vector = input_vector.reshape(input_vector.shape[1])
        delta, _ = self.sigmoid(np.dot(output_words, input_vector) * directions)
        delta_minus = (delta - 1) * directions
        cost = -np.sum(np.log(delta))
        grad_pred = np.dot(delta_minus.reshape(1,K+1), output_words).flatten()
        grad_min = np.dot(delta_minus.reshape(K+1,1), input_vector.reshape(1,N))
        for k in range(K+1):
            grad[indices[k]] += grad_min[k,:]
        return cost, grad_pred, grad
    

    def cbow(self,current_word, C, context_words, tokens, input_vectors0,
             output_vectors, word2vec_cost_grads):
        cost = 0.0
        grad_in = np.zeros(input_vectors0.shape) 
        grad_out = np.zeros(output_vectors.shape)
        input_vectors=np.zeros((input_vectors0.shape[0],1))
        #print(input_vectors)
        #print(input_vectors.shape)
        try:
            idx = tokens[current_word]
        except:
            pass
        for context in context_words:
            try:
                input_vectors += input_vectors0[:,tokens[context]].reshape(-1, 1)
            except:
                pass
        #print(input_vectors)
        #print(input_vectors.shape)
        input_vectors = input_vectors.reshape(1, input_vectors.shape[0])
        try:
                dcost, g_in, g_out = word2vec_cost_grads(input_vectors,
                                                        output_vectors,idx
                                                        ,
                                                        )
                cost += dcost
                grad_in[:,idx] += g_in
                grad_out += g_out
        except:
                pass
        return cost, grad_in, grad_out


    

    def skipgram(self,current_word, C, context_words, tokens, input_vectors,
             output_vectors, word2vec_cost_grads):
        cost = 0.0
        grad_in = np.zeros(input_vectors.shape) 
        grad_out = np.zeros(output_vectors.shape)
        try:
            idx = tokens[current_word]
        except:
            pass
        for context in context_words:
            try:
                input_vector = input_vectors[:,idx]
                input_vector = input_vector.reshape(1, input_vector.shape[0])
                dcost, g_in, g_out = word2vec_cost_grads(input_vector,
                                                        output_vectors,
                                                        tokens[context],
                                                        )
                cost += dcost
                grad_in[:,idx] += g_in
                grad_out += g_out
            except:
                pass
        return cost, grad_in, grad_out

    def getRandomContext(self, C=5):
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)
        context = sent[max(0, wordID - C):wordID]
        if wordID+1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)]
        centerword = sent[wordID]
        context = [w for w in context if w != centerword]
        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)

    
    def word2vec_sgd_wrapper(self,word2vec_model, tokens, word_vectors, C,
                         word2vec_gradient):
        batchsize = 50
        cost = 0.0
        grad = np.zeros(word_vectors.shape) 
        m = word_vectors.shape[0]
        input_vectors = word_vectors[:int(m/2),:].T
        output_vectors = word_vectors[int(m/2):,]
        for i in range(batchsize):
            C1 = random.randint(1,C)
            centerword, context = self.getRandomContext(C1) # Example of output: ('c', ['a', 'b', 'e'])
            c, gin, gout = word2vec_model(centerword, C1, context, tokens,
                                        input_vectors, output_vectors,
                                        word2vec_gradient)
            cost += c / batchsize
            grad[:int(m/2),:] += gin.T / batchsize
            grad[int(m/2):,] += gout / batchsize
        return cost, grad


    def sgd(self,f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=100):
        ANNEAL_EVERY = 20000
        if useSaved:
            start_iter, oldx, state = self.load_saved_params()
            if start_iter > 0:
                x0 = oldx
                step *= 0.5 ** (start_iter / ANNEAL_EVERY)
            if state:
                random.setstate(state)
        else:
            start_iter = 0
        x = x0
        if not postprocessing:
            postprocessing = lambda x: x
        expcost = None
        for iter in range(start_iter + 1, iterations + 1):
            cost, grad = f(x)
            x = x - step * grad
            if iter % PRINT_EVERY == 0:
                if not expcost:
                    expcost = cost
                else:
                    expcost = .95 * expcost + .05 * cost
                print("iter %d: %f" % (iter, expcost))
            if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
                self.save_params(iter, x)
            if iter % ANNEAL_EVERY == 0:
                step *= 0.5
        return x
    

    def initialize(self):
        self.n_words=len(self.vocab)
        word_vectors = np.concatenate(((np.random.rand(self.n_words, self.dim_vectors) - .5) /  self.dim_vectors, np.zeros((self.n_words, self.dim_vectors))), axis=0)
        word_vectors0 = self.sgd(lambda vec: self.word2vec_sgd_wrapper(self.skipgram, self.tokens_id, vec, self.window_size, self.negative_sampling),
                            word_vectors, 0.3, 30000, None, True, PRINT_EVERY=100)
        return word_vectors0
        


obj=Word2vec()
obj.load_data_sentences()
print("No of sentences: ",len(obj.sentences))
obj.tokens()
n_words=len(obj.tokens_id)
print("No of Tokens: ",n_words)
wordvectors=obj.initialize()
print("No of Tokens: ",obj.n_words)
toc = time.process_time()
print("Training time: " + str(toc-tic))
print("Vector shape: ",wordvectors.shape)
print("Vector shape: ",wordvectors[:obj.n_words,:].shape)
print("Vector shape: ",wordvectors[obj.n_words:,:].shape)

wordvectors = (wordvectors[:obj.n_words,:] + wordvectors[obj.n_words:,:])
visualizeWords = ["life", "death", "promising", 
	"good", "great", "cool",  "wonderful", "well", "amazing",
	"worth", "sweet", "enjoyable", "bad", "waste", 
	"annoying"]

visualizeIdx = [obj.tokens_id[word] for word in visualizeWords]
visualizeVecs = wordvectors[visualizeIdx, :]
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


