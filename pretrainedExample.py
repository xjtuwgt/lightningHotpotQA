from jdscripts.pretrainedencoders import load_model


model = load_model(model_type='roberta')

for name, param in model.named_parameters():
    print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))