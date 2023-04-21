import torch
from transformers import GPT2Tokenizer, GPT2Model

device = torch.device("cuda:1") # if a GPU is available

tokenizer = GPT2Tokenizer.from_pretrained('./tokenizer')
model = GPT2Model.from_pretrained('./model')

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)



"""
prompt = "<|endoftext|>Fluoroacetate Dehalogenase"
device = torch.device("cuda:1") # if a GPU is available
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
model = GPT2LMHeadModel.from_pretrained('./model').to(device)

input_ids = tokenizer.encode(prompt,return_tensors='pt').to(device)
# change max_length or num_return_sequences to your requirements
output = model.generate(input_ids, top_k=950, repetition_penalty=1.2, max_length=300,
                        eos_token_id=1,pad_token_id=0,do_sample=True, num_return_sequences=10) #100)

"""
#print(output)


"""
with open(f'prompt{prompt}_GPT2-sm.fasta', 'w') as f:
    for i, seq in enumerate(output):
        seq = tokenizer.decode(output[i])                                                                                                                                                     
        write_seq = seq.replace(' ', '').replace('<pad>', '').replace('<sep>', '').replace('<start>', '').replace(prompt, '').replace('<|endoftext|>', '').replace('<end>', '')
        f.write(f'>prompt{prompt}_GT2-sm_{i}\n')
        f.write(write_seq + '\n')

"""
