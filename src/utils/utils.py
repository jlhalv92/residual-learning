def copy_weights(source_model, target_model):
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
        if source_param.data.shape == target_param.data.shape:
            target_param.data.copy_(source_param.data)

def set_weights_to_zero(model):
    for param in model.parameters():
        param.data.fill_(0)

def set_bias_to_zero(model):
    for param in model.parameters():
        if param.dim() > 1:
            # If the parameter is a bias term (not a weight matrix),
            # set it to zero
            param.data.zero_()

def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False