from gensim.models import KeyedVectors
import re
import sys
import argparse
import os
import subprocess
class related_word_model():
    def __init__(self) -> None:
        self.model=KeyedVectors.load('model1/malayalam_gensim_word2vec_model_dim200_win25_ns15_nse1_seed10_wv')
        self.parser=argparse.ArgumentParser(description='-word words_to_find -topn  no_of _related _words _to _extraxt (\nEg: -word Animal -topn 5)  (Eg: -word "Animal man woman" -topn 5)')
        self.parser.add_argument('-word', type=str, help='words to find related words for.')
        self.parser.add_argument('-topn',type=int,help='no of similar words to find default is 10',default=10)
        self.parser.add_argument('-file',type=str,help='filename to save the results. Default is Related_words.txt',default='Related_words.txt')
        self.parser.add_argument('-append',type=int,help='append to existing Related_words.txt (1 for yes 0 for no) Default is 1',default=1)

   
    def main(self):
    
        args=self.parser.parse_args()
        if  args.word is None :
            print('No arguments passed')
            self.parser.print_help()
            return
        filename=args.file
        if '.txt' not in filename:
            filename+='.txt'
        with open(filename,['w','a'][args.append],encoding='utf-8') as file:
            for word in args.word.split():
                file.write(f'Related words for {word}\n')
                try:
                    [file.write(f'{x[0]}\n') for x in self.model.most_similar(re.sub(r'[^\u0D00-\u0D7F\s]', '', word),topn=args.topn)]
                except:
                    print('The word is out of vocab.')
                    return
                file.write('\n\n\n')
        print(filename)
        try:
            subprocess.Popen([r"C:\Users\Administrator\AppData\Local\Programs\Microsoft VS Code\Code.exe", '--reuse-window', filename])
        except:
            os.startfile(filename)

if __name__=='__main__':
    related_word_model().main()