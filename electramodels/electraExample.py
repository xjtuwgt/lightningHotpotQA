from model_envs import MODEL_CLASSES
import torch
model_name = "google/electra-small-discriminator"

config_class, model_class, tokenizer_class = MODEL_CLASSES['electra']

model = model_class.from_pretrained("google/electra-small-discriminator")
tokenizer = tokenizer_class.from_pretrained("google/electra-small-discriminator")

print(tokenizer.cls_token, tokenizer.sep_token)
# print(tokenizer)
# #
# # sentence = "The quick brown fox jumps over the lazy dog"
fake_sentence = "The Quick brown Fox fake over the LLLLLazy dog"
#
fake_tokens = tokenizer.tokenize(fake_sentence)
print(fake_tokens)
# print(fake_tokens)
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
# # fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
#
# inputs = {'input_ids':      input_ids,
#                   'attention_mask': None,
#                   'token_type_ids': None}
# print(type(input_ids), input_ids.shape)
#
# # print(fake_inputs)
# discriminator_outputs = model(**inputs)
# print(discriminator_outputs.last_hidden_state.shape)
# print(discriminator_outputs)
# predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
#
# [print("%7s" % token, end="") for token in fake_tokens]
#
# [print("%7s" % int(prediction), end="") for prediction in predictions.tolist()]

import os

