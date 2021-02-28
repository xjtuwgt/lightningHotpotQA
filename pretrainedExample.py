from jdscripts.pretrainedencoders import load_model, save_model_as_pkl, load_hotpotqa_model, save_hotpotqa_model
from jdscripts.pretrainedencoders import load_model_with_enconder

# roberta_model_name = 'ahotrod/roberta_large_squad2'
# albert_model_name = 'mfeb/albert-xxlarge-v2-squad2'
# model = load_model(model_type='roberta', model_name=roberta_model_name)
# for name, param in model.named_parameters():
#     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
# save_model_as_pkl(encoder=model, model_name=roberta_model_name)
#
# model = load_model(model_type='albert', model_name=albert_model_name)
# for name, param in model.named_parameters():
#     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
# save_model_as_pkl(encoder=model, model_name=albert_model_name)
#
# # model = load_model(model_type='albert', model_name=albert_model_name)
# #
# # for name, param in model.named_parameters():
# #     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
# model_type = 'roberta'
# model_name = 'roberta-large'
# model = load_hotpotqa_model(model_type=model_type, model_name=model_name)
# for name, param in model.named_parameters():
#     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
# save_hotpotqa_model(encoder=model, model_type=model_type, model_name=model_name)

model_type = 'roberta'
model_name = 'roberta-large'
encoder_name = 'ahotrod/roberta_large_hotpotqa'
model = load_model_with_enconder(model_type=model_type, model_name=model_name, encoder_model_name=encoder_name)