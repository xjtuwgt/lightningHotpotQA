from jdscripts.pretrainedencoders import load_model

roberta_model_name = 'ahotrod/roberta_large_squad2'
albert_model_name = 'mfeb/albert-xxlarge-v2-squad2'
# model = load_model(model_type='roberta', model_name=roberta_model_name)

model = load_model(model_type='albert', model_name=albert_model_name)

for name, param in model.named_parameters():
    print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))