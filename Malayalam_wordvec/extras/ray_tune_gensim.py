import re
import gensim.models
from gensim.models.callbacks import CallbackAny2Vec
import copy
from ray import tune
from ray.train import RunConfig
import json
from ray import train

class Model(): 
    def train(self,config):
        results=[]
        model= gensim.models.Word2Vec(sentences=config['sentences'],min_count=1,
                                       vector_size=config['vector_size'],window=config['window'],sg=1,negative=10,
                                       ns_exponent =0,seed=10,workers=10)
        results.append(f' Model params: Epochs: {model.epochs}, vector size : {model.vector_size}, window size: {model.window}')
        for word in config['words']:
            result=model.wv.most_similar(word)
            for related_word in result:
                results.append(f'{word:<20} : {related_word[0]:<20}')
        with open('Z:/My Folders/Cleeto_Tasks/task5/word2vec_model_10_epochs_results.txt','a',encoding='utf-8') as file:
            for line in results:
                file.write(line+'\n')
      
      
        

class RayTuner():
    
    def load_data(self):
        self.sentences=[]
        with open('malayalam_senteces2.txt','r',encoding='utf-8') as file:
                    for line in file:
                        line = re.sub(r'[^\u0D00-\u0D7F\s]', ' ', line)
                        self.sentences += [[w for w in line.split()]]

    
    def main(self):
        self.load_data()
        self.words=[re.sub(r'[^\u0D00-\u0D7F\s]', '', 'ആള്‍'),re.sub(r'[^\u0D00-\u0D7F\s]', '', 'സഹോദരന്‍'),
                    re.sub(r'[^\u0D00-\u0D7F\s]', '', 'ദൈവം')]
        config={'vector_size':tune.choice([100,200,300,400]),'window':tune.choice([10,15,20,25]),
                'sentences':self.sentences,'words':self.words,}
        result=tune.Tuner(Model().train,
            param_space=config,
            tune_config=tune.TuneConfig(
            num_samples=25),
            run_config=RunConfig(
        name="experiment_name",
        storage_path="~/ray_results/",
    )    )
        result=result.fit()
    


if __name__=='__main__':
      RayTuner().main()