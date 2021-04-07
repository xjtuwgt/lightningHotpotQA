import torch
from jdscripts.pretrainedencoders import load_model, save_model_as_pkl, load_hotpotqa_model, save_hotpotqa_model
from jdscripts.pretrainedencoders import load_model_with_enconder, load_hgn_hotpotqa_model, save_hgn_hotpotqa_model
from jdscripts.pretrainedencoders import model_intialization_test

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
# model_type = 'roberta'
# model_name = 'roberta-large'
# model = load_hotpotqa_model(model_type=model_type, model_name=model_name)
# for name, param in model.named_parameters():
#     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
# save_hotpotqa_model(encoder=model, model_type=model_type, model_name=model_name)
#
# model_type = 'albert'
# model_name = 'albert-xxlarge-v2'
# model = load_hotpotqa_model(model_type=model_type, model_name=model_name)
# for name, param in model.named_parameters():
#     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
# save_hotpotqa_model(encoder=model, model_type=model_type, model_name=model_name)

# model_type = 'albert'
# model_name = 'albert-xxlarge-v2'
# prtrained_name = 'albert-xxlarge-v2_hotpotqa'
# model = model_intialization_test(model_type=model_type, model_name=model_name, petrained_name=prtrained_name)
# for name, param in model.named_parameters():
#     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))

# model_type = 'roberta'
# model_name = 'roberta-large'
# encoder_name = 'roberta/roberta-large_hotpotqa'
# # model = load_model_with_enconder(model_type=model_type, model_name=model_name, encoder_model_name=encoder_name)

# model_type = 'roberta'
# model_name = 'roberta-large'
# exp_name = 'train.roberta.bs2.lr1e-05.datahgn_low.seed303'
# encoder_pickle_name = 'encoder_5.pkl'
# model = load_hgn_hotpotqa_model(model_type=model_type, model_name=model_name, exp_name=exp_name, encoder_pkl_name=encoder_pickle_name)
# for name, param in model.named_parameters():
#     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
# save_hgn_hotpotqa_model(encoder=model, model_type=model_type, model_name=model_name)
#
# model_type = 'roberta'
# model_name = 'roberta-large'
# prtrained_name = 'roberta-large_hgn'
# model = model_intialization_test(model_type=model_type, model_name=model_name, petrained_name=prtrained_name)
# for name, param in model.named_parameters():
#     print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))

electra_model_name = 'mrm8488/electra-large-finetuned-squadv1'
model = load_model(model_type='electra', model_name=electra_model_name)
for name, param in model.named_parameters():
    print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
save_model_as_pkl(encoder=model, model_name=electra_model_name)

model_type = 'electra'
model_name = 'google/electra-large-discriminator'
model = model_intialization_test(model_type=model_type, model_name=model_name, petrained_name=electra_model_name)
for name, param in model.named_parameters():
    print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))