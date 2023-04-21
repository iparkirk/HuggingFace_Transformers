from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
# save
tokenizer.save_pretrained('./tokenizer')
model.save_pretrained('./model')
#text = "Replace me by any text you'd like."
#encoded_input = tokenizer(text, return_tensors='pt')
#output = model(**encoded_input)

