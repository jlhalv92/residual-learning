import torch
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

def compare_model_parameters(model1, model2):
    params1 = model1.state_dict()
    params2 = model2.state_dict()

    for key1, value1 in params1.items():
        if key1 in params2:
            value2 = params2[key1]
            if torch.equal(value1, value2):
                print(f"Parameters for {key1} are the same.")
            else:
                print(f"Parameters for {key1} are different.")
        else:
            print(f"{key1} is not present in the second model.")

    for key2 in params2.keys():
        if key2 not in params1:
            print(f"{key2} is not present in the first model.")