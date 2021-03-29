from model_envs import MODEL_CLASSES

model_name = "google/electra-small-discriminator"

config_class, model_class, tokenizer_class = MODEL_CLASSES['electra']

model = model_class.from_pretrained("google/electra-small-discriminator")
tokenizer = tokenizer_class.from_pretrained("google/electra-small-discriminator")
# print(tokenizer)
# #
# # sentence = "The quick brown fox jumps over the lazy dog"
fake_sentence = "The quick brown fox fake over the lazy dog"
#
fake_tokens = tokenizer.tokenize(fake_sentence)
print(fake_tokens)
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
print(fake_inputs)
discriminator_outputs = model(fake_inputs)
print(discriminator_outputs)
# predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
#
# [print("%7s" % token, end="") for token in fake_tokens]
#
# [print("%7s" % int(prediction), end="") for prediction in predictions.tolist()]

import os

