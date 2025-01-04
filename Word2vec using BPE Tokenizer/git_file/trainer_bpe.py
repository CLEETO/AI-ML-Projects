import random
import time
from stanford_bpe import *
import sys
from word2vec import *
from stochastic_gradient_descent import *

tic = time.process_time()

random.seed(314, version=1)
dataset = StanfordSentiment()
tokens = dataset.tokens()
n_words = len(tokens)
print(n_words)
print(max(tokens.values()))
#n_words=max(tokens.values())

print("Number of words is equal to: " + str(len(tokens)))

# Train 10-dimensional vectors
dim_vectors = 10

# Context size
C = 5

random.seed(31415, version=1)
np.random.seed(9265)

word_vectors = np.concatenate(((np.random.rand(n_words, dim_vectors) - .5) / \
	dim_vectors, np.zeros((n_words, dim_vectors))), axis=0)

word_vectors0 = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negative_sampling),
    word_vectors, 0.3, 30000, None, True, PRINT_EVERY=10)

print("Sanity check: cost at convergence should be around or below 10")

toc = time.process_time()

print("Training time: " + str(toc-tic))

# Visualize the word vectors you trained
_, wordVectors0, _ = load_saved_params()
#print(word_vectors.shape)
#print(wordVectors0.shape)
word_vectors = (wordVectors0[:n_words,:] + wordVectors0[n_words:,:])
#print("DScdscsc",word_vectors.shape)
#print(tokens.keys())
# Binary strings defined in Python 3.
# visualizeWords = ["the", "a", "an", ",",".","?","!", "``", "''", "--",
# 	"good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
# 	"worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb",
# 	"annoying", 'immaculately']

visualizeWords = [
     " peak", " cool", " trash", " rock", " solid", " space", " garbage", " worst", " death-defying"
]
def find_embedding(word):
    #search_word_embedding=dataset._tokenizer.encode(word).tokens
    search_word_embedding=dataset._tokenizer.tokenize(word)
    print(search_word_embedding)
    #print(search_word_embedding)
    #search_word_embedding=dataset._tokenizer.decode(search_word_embedding)
    #print(search_word_embedding)
    search_word_vector=np.zeros(word_vectors.shape[1]).tolist()
    
    for embedding in search_word_embedding:
        try:
            search_word_vector+=word_vectors[tokens[embedding]]
        except:
         print('Exception: ',embedding)
        #print(search_word_vector)
    search_word_vector/=len(search_word_embedding)
    
         #pass
    return search_word_vector
'''
word1,word2=['american','indian']
word1vector=find_embedding(word1)
word2vector=find_embedding(word2)
#print(word1vector,word2vector)
cosine_similarity=np.dot(word1vector,word2vector)/(np.linalg.norm(word1vector)*np.linalg.norm(word2vector))
print(cosine_similarity)

search_word='masterpiece'
search_word_vector=find_embedding(search_word)

#print(dataset._tokenizer.tokenize('The film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game .'))
words=[]
with open("datasets/stanford-sentiment-tree-bank/dataset_sentences.txt", "r") as f:
            first = True
            for line in f:

                if first:
                    first = False
                    continue
                splitted = line.strip().split()[1:]
                #splitted=' '.join(splitted)
                #print(splitted)
                #splitted = self._tokenizer.encode(splitted.strip().lower().replace('-lrb-', '(').replace('-rrb-', ')').replace('ã©','é'))
                #splitted = self._tokenizer.tokenize(splitted.strip().lower().replace('-lrb-', '(').replace('-rrb-', ')').replace('ã©','é'))
                
                #print(splitted)
                #break
                #print(splitted)
                # Deal with some peculiar encoding issues with this file
                # sentences += [[w.lower().decode("utf-8").encode('latin1') for w in splitted]] -- Works only in Python 2
                #sentences += [[w.lower() for w in splitted]]
                #sentences += [[w for w in splitted.tokens]]
                for word in splitted:
                     if word.lower() not in words:
                          words.append(word.lower())
#print(words)
for word in words:
     word_embedding=find_embedding(word)
     cosine_similarity=np.dot(search_word_vector,word_embedding)/(np.linalg.norm(search_word_vector)*np.linalg.norm(word_embedding))
     if cosine_similarity>0.999:
          print(word)
'''
visualizeVecs=[]
for word in visualizeWords:
    
    embedding=find_embedding(word)
    #print(embedding)
    visualizeVecs+=[embedding]

        
#visualizeIdx = [tokens[word] for word in visualizeWords]

#visualizeVecs = wordVectors[visualizeIdx, :]
#print(len(visualizeVecs),len(visualizeWords))
#print(visualizeVecs)
#print(visualizeVecs)
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))

covariance = 1.0 / len(visualizeVecs) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])
coord=coord
for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
    	bbox=dict(facecolor='green', alpha=0.1))



plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors.png')
plt.show()
