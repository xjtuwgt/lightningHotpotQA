from transformers import ElectraModel, ElectraConfig
from transformers import ElectraTokenizer, ElectraModel
from transformers import ElectraTokenizer, ElectraForQuestionAnswering
import torch

electra_large_model_name = 'google/electra-large-discriminator'

tokenizer = ElectraTokenizer.from_pretrained(electra_large_model_name)
model = ElectraModel.from_pretrained(electra_large_model_name, return_dict=True)
for name, param in model.named_parameters():
    print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

print(outputs)


# tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
# model = ElectraForQuestionAnswering.from_pretrained('google/electra-small-discriminator', return_dict=True)
#
# question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
# inputs = tokenizer(question, text, return_tensors='pt')
# start_positions = torch.tensor([1])
# end_positions = torch.tensor([3])
#
# outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
# loss = outputs.loss
# start_scores = outputs.start_logits
# end_scores = outputs.end_logits