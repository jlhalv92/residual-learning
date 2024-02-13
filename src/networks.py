import torch.nn as nn
import torch
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(CriticNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


# class ResidualBlock(nn.Module):
#     def __init__(self, n_features, **kwargs):
#         super(ResidualBlock, self).__init__()
#
#         self.layer_1 = nn.Linear(n_features, n_features)
#         self.layer_2 = nn.Linear(n_features, n_features)
#
#         nn.init.xavier_uniform_(self.layer_1.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self.layer_2.weight,
#                                 gain=nn.init.calculate_gain('linear'))
#
#     def forward(self, x):
#         identity = x
#         features1 = F.relu(self.layer_1(x))
#         features2 = self.layer_2(features1)
#         features2 += identity
#         return F.relu(features2)



class ResidualBlock(nn.Module):
    def __init__(self, n_features, **kwargs):
        super(ResidualBlock, self).__init__()

        self.layer_2 = nn.Linear(n_features, n_features)

        nn.init.xavier_uniform_(self.layer_2.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x):

        identity = x
        features2 = self.layer_2(x)
        features2 += identity

        return F.relu(features2)



class ResidualBlock2(nn.Module):
    def __init__(self, n_features, **kwargs):
        super(ResidualBlock2, self).__init__()

        self.layer_2 = nn.Linear(n_features, n_features)
        self.layer_1 = nn.Linear(n_features, n_features)

        nn.init.xavier_uniform_(self.layer_1.weight,
                                gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.layer_2.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x):

        identity = x
        features1=F.relu(self.layer_1(x))
        features2 = self.layer_2(features1)
        features2 += identity

        return F.relu(features2)


class LinearOutput(nn.Module):
    def __init__(self, n_features,out, **kwargs):
        super(LinearOutput, self).__init__()

        self.layer_out = nn.Linear(n_features, out)

        nn.init.xavier_uniform_(self.layer_out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        out = self.layer_out(x)
        return out

class CriticBoostedNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(CriticBoostedNetwork, self).__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features
        self._h1 = nn.Linear(n_input, n_features)

        self._rho_0 = ResidualBlock(n_features)
        self._out = LinearOutput(n_features, self.n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))


    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = self._rho_0(features1)
        q = self._out(features2)

        return torch.squeeze(q)

    def freeze(self):
        # Iterate through all parameters in the model and set requires_grad to False
        for param in self.parameters():
            param.requires_grad = False

    def print_frozen_layers(self):
        frozen_layers = [name.split(".")[0] for name, param in self.named_parameters() if not param.requires_grad]
        print("Frozen Layers:", list(dict.fromkeys(frozen_layers)))

    def add_residual(self, residual_id):
        new_layer = ResidualBlock(self.n_features)
        new_layer = new_layer.to(next(self.parameters()).device)
        self.add_module(residual_id, new_layer)

    def add_output(self):
        out_layer = LinearOutput(self.n_features, self.n_output)
        out_layer = out_layer.to(next(self.parameters()).device)
        self._out = out_layer

    def remove_last_layer(self):
        # Get the names of all modules (layers) in the model
        all_module_names = list(self._modules.keys())

        # Remove the last layer
        last_layer_name = all_module_names[-1]
        delattr(self, last_layer_name)


class Q0(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q0, self).__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features
        self._h1 = nn.Linear(n_input, n_features)
        self._rho_0 = ResidualBlock(n_features)
        self._out = LinearOutput(n_features, self.n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = self._rho_0(features1)
        q = self._out(features2)

        return torch.squeeze(q)


class Q1(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q1, self).__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features
        self._h1 = nn.Linear(n_input, n_features)

        self._rho_0 = ResidualBlock(n_features)
        self._rho_1 = ResidualBlock(n_features)
        self._out = LinearOutput(n_features, self.n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))


    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = self._rho_0(features1)
        features3 = self._rho_1(features2)
        q = self._out(features3)

        return torch.squeeze(q)



class Q2(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q2, self).__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features
        self._h1 = nn.Linear(n_input, n_features)

        self._rho_0 = ResidualBlock(n_features)
        self._rho_1 = ResidualBlock(n_features)
        self._rho_2 = ResidualBlock(n_features)
        self._out = LinearOutput(n_features, self.n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))


    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = self._rho_0(features1)
        features3 = self._rho_1(features2)
        features4 = self._rho_2(features3)
        q = self._out(features4)

        return torch.squeeze(q)


class ResidualCritic(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ResidualCritic, self).__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features

        self.layer_1 = nn.Linear(n_input, n_features)
        self.layer_2 = nn.Linear(n_features, n_features)
        self.out = nn.Linear(n_features, self.n_output)

        nn.init.xavier_uniform_(self.layer_1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer_2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.out.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        identity = x
        out = F.relu(self.layer_1(x))
        out = self.layer_2(out)
        out = F.relu(out + identity)
        q = self.out(out)
        return torch.squeeze(q)



class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, two_layers=False, **kwargs):
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape
        self._n_games = len(self._n_input)

        self._h1 = nn.Linear(self._n_input[0], n_features)
        if two_layers:
            self._h2 = nn.Linear(n_features, n_features)
        else:
            self._h2 = None
        self._q = nn.Linear(n_features, self._n_output[0])

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        if two_layers:
            nn.init.xavier_uniform_(self._h2.weight,
                                    gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._q.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        state = state.float()
        h = F.relu(self._h1(state))
        if self._h2 is not None:
            h = F.relu(self._h2(h))
        q = self._q(h)

        if action is not None:
            action = action.long()
            q = torch.squeeze(q.gather(1, action))

        return q