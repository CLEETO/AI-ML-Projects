import os.path as op
import _pickle as pickle
import random
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
SAVE_PARAMS_EVERY = 1000
REGULARIZATION = [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

class sentiments:

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
        

    

    

# Replace characters in the text
        for incorrect, correct in replacements.items():
            text = text.replace(incorrect, correct)

        return text


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
        self.num_of_sent=0
        for line in self.sentences:
            self.num_of_sent+=1
            for word in line:
                self.wordcount+=1
                if word not in self.vocab:
                    self.tokens_id[word]=idx
                    self.token_freq[word]=1
                    self.vocab.append(word)
                    idx+=1
                else:
                    self.token_freq[word]+=1


    def sentiment_labels(self):
        if hasattr(self, "_sentiment_labels") and self._sentiment_labels:
            return self._sentiment_labels
        dictionary = dict()
        phrases = 0
        with open("dictionary.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                index=splitted[1]
                splitted=splitted[0].lower()
                splitted=splitted.strip().lower().replace('-lrb-', '(').replace('-rrb-', ')')
                #splitted=self.clean(splitted)
                dictionary[splitted] = int(index)
                phrases += 1
        labels = [0.0] * phrases
        with open("sentiment_labels.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])
        sentiment_labels = [0.0] * self.num_of_sent
        sentences = self.sentences
        self.count=0
        for i in range(self.num_of_sent):
            sentence = sentences[i]
            full_sent = " ".join(sentence)
            try:
                
                sentiment_labels[i] = labels[dictionary[full_sent]]
                self.count=self.count+1
                #print(i,sentiment_labels[i])
            except KeyError:
                 
                 print("I got a KeyError - reason '%s'" % str(full_sent))
        self._sentiment_labels = sentiment_labels
        print('Matched ',self.count,'out of ',len(self.sentences))
        return self._sentiment_labels


    def dataset_split(self):
        if hasattr(self, "_split") and self._split:
            return self._split
        split = [[] for i in range(3)]
        with open("dataset_split.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]

        self._split = split
        #print('completed')
        return self._split
    
    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sentiment_labels()[sentId])

    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4


    def getDevSentences(self):
        return self.getSplitSentences(2)


    def getTestSentences(self):
        return self.getSplitSentences(1)


    def getTrainSentences(self):
        return self.getSplitSentences(0)


    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split()
        return [(self.sentences[i], self.categorify(self.sentiment_labels()[i])) for i in ds_split[split]]

    def getSentenceFeature(self,tokens, word_vectors, sentence):
        sentence_vector = np.zeros((word_vectors.shape[1],))
        for word in sentence:
            vector = word_vectors[tokens[word],:]
            sentence_vector += vector

        sentence_vector /= len(sentence)

        return sentence_vector
                
    def save_params(self,iter, params):
        with open("saved_params_%d.npy" % iter, "wb") as f:
            pickle.dump(params, f)
            pickle.dump(random.getstate(), f)
    
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

    
    def softmax_cost_grads_reg(self,features, labels, weights, regularization = 0.0, nopredictions = False):
    
        probabilities, _ = self.softmax(features.dot(weights).T)

        if len(features.shape) > 1:
            N = features.shape[0]
        else:
            N = 1

        # A vectorized implementation of 1/N * sum(cross_entropy(x_i, y_i)) + 1/2*|w|^2

        cost = np.sum(-np.log(probabilities[labels, range(N)])) / N
        cost += 0.5 * regularization * np.sum(weights ** 2)

        grad = np.zeros_like(weights)
        pred = 0;

        numlabels = np.shape(weights)[1]

        delta = probabilities.T - np.eye(numlabels)[labels]
        grad = (np.dot(features.T,delta) / N) + regularization*weights

        if N > 1:
            pred = np.argmax(probabilities, axis=0)
        else:
            pred = np.argmax(probabilities)

        if nopredictions:
            return cost, grad
        else:
            return cost, grad, pred
    
    def softmax_wrapper(self,features, labels, weights, regularization = 0.0):
        cost, grad, _ = self.softmax_cost_grads_reg(features, labels, weights, regularization)
        return cost, grad
    
    def accuracy(self,y, yhat):
        print(y.shape)
        print(yhat.shape)
        assert(y.shape == yhat.T.shape)
        return np.sum(y == yhat) * 100.0 / y.size
        
    

dataset = sentiments()
dataset.load_data_sentences()
print("No of sentences: ",len(dataset.sentences))
dataset.tokens()
n_words = len(dataset.tokens_id)
print("Nof tokens in vocab: ",n_words)
_, wordVectors0, _ = dataset.load_saved_params()
print("Vector Shape: ",wordVectors0.shape)
wordVectors = (wordVectors0[:n_words,:] + wordVectors0[n_words:,:])
dimVectors = wordVectors.shape[1]   
trainset = dataset.getTrainSentences()
print('completed')
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)
#print(nTrain)
for i in range(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = dataset.getSentenceFeature(dataset.tokens_id, wordVectors, words)
devset = dataset.getDevSentences()
print('completed')
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in range(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = dataset.getSentenceFeature(dataset.tokens_id, wordVectors, words)
print('completed')
results = []
for regularization in REGULARIZATION:
    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dimVectors, 5)
    print("Training for reg=%f" % regularization)

    # We will do batch optimization
    weights = dataset.sgd(lambda weights: dataset.softmax_wrapper(trainFeatures, trainLabels,
        weights, regularization), weights, 3.0, 1000, PRINT_EVERY=100)

    # Test on train set
    _, _, pred = dataset.softmax_cost_grads_reg(trainFeatures, trainLabels, weights)
    trainAccuracy = dataset.accuracy(trainLabels, pred)
    print("Train accuracy (%%): %f" % trainAccuracy)

    # Test on dev set
    _, _, pred = dataset.softmax_cost_grads_reg(devFeatures, devLabels, weights)
    devAccuracy = dataset.accuracy(devLabels, pred)
    print("Dev accuracy (%%): %f" % devAccuracy)
    
    # Save the results and weights
    results.append({
        "reg" : regularization,
        "weights" : weights,
        "train" : trainAccuracy,
        "dev" : devAccuracy})

# Print the accuracies

print("=== Recap ===")
print("Reg\t\tTrain\t\tDev")
for result in results:
    print("%E\t%f\t%f" % (
        result["reg"],
        result["train"],
        result["dev"]))

# Pick the best regularization parameters
BEST_REGULARIZATION = None
BEST_WEIGHTS = None

bestdev = 0
for result in results:
    if result["dev"] > bestdev:
        BEST_REGULARIZATION = result["reg"]
        BEST_WEIGHTS = result["weights"]
        bestdev = result["dev"]
with open("saved_params_reg%d.npy" % BEST_REGULARIZATION, "wb") as f:
        pickle.dump(BEST_WEIGHTS, f)
        pickle.dump(random.getstate(), f)
# Test regularization results on the test set
testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in range(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = dataset.getSentenceFeature(dataset.tokens_id, wordVectors, words)

_, _, pred = dataset.softmax_cost_grads_reg(testFeatures, testLabels, BEST_WEIGHTS)
print("Best regularization value: %E" % BEST_REGULARIZATION)
print("Test accuracy (%%): %f" % dataset.accuracy(testLabels, pred))

# Make a plot of regularization vs accuracy
plt.plot(REGULARIZATION, [x["train"] for x in results])
plt.plot(REGULARIZATION, [x["dev"] for x in results])
plt.xscale('log')
plt.xlabel("regularization")
plt.ylabel("accuracy")
plt.legend(['train', 'dev'], loc='upper left')
plt.savefig("regularization-accuracy_img.png")
plt.show()
