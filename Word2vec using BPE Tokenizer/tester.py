from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer2/tokenizer.json")
while True:
    inp=input("Enter: ")
    print(tokenizer.tokenize(inp))