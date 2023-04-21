from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2Model.from_pretrained('gpt2-large')
## save
tokenizer.save_pretrained('./tokenizer')
model.save_pretrained('./model')

#text = "Replace me by any text you'd like."
#encoded_input = tokenizer(text, return_tensors='pt')
#output = model(**encoded_input)