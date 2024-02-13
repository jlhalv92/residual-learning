import torch.nn as nn
import torch
import torch.nn.functional as F
from src.networks import ResidualBlock, LinearOutput, ResidualBlock2


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

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

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted

class Network_Q1(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        q = self._h4(features3)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted

class Network_Q2(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.Linear(n_features, n_features)
        self._h5= nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        features4 = F.relu(self._h4(features3))
        q = self._h5(features4)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted

class Network_Q3(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.Linear(n_features, n_features)
        self._h5 = nn.Linear(n_features, n_features)
        self._h6 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self._h6.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        features4 = F.relu(self._h4(features3))
        features5 = F.relu(self._h5(features4))
        q = self._h6(features5)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted

class Network_Q4(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.Linear(n_features, n_features)
        self._h5 = nn.Linear(n_features, n_features)
        self._h6 = nn.Linear(n_features, n_features)
        self._h7 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h6.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h7.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        features4 = F.relu(self._h4(features3))
        features5 = F.relu(self._h5(features4))
        features6 = F.relu(self._h6(features5))
        q = self._h7(features6)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted



class Network_Q5(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.Linear(n_features, n_features)
        self._h5 = nn.Linear(n_features, n_features)
        self._h6 = nn.Linear(n_features, n_features)
        self._h7 = nn.Linear(n_features, n_features)
        self._h8 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h6.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h7.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h8.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        features4 = F.relu(self._h4(features3))
        features5 = F.relu(self._h5(features4))
        features6 = F.relu(self._h6(features5))
        features7 = F.relu(self._h7(features6))
        q = self._h8(features7)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


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

    def forward(self,state, action=None):

        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._rho_0(features1)
        q = self._out(features2)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


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

    def forward(self,state, action=None):

        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._rho_0(features1)
        features3 = self._rho_1(features2)


        q = self._out(features3)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


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

    def forward(self,state, action=None):

        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._rho_0(features1)
        features3 = self._rho_1(features2)
        features4 = self._rho_2(features3)

        q = self._out(features4)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


class Q3(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q3, self).__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features

        self._h1 = nn.Linear(n_input, n_features)
        self._rho_0 = ResidualBlock(n_features)
        self._rho_1 = ResidualBlock(n_features)
        self._rho_2 = ResidualBlock(n_features)
        self._rho_3 = ResidualBlock(n_features)
        self._out = LinearOutput(n_features, self.n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self,state, action=None):

        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._rho_0(features1)
        features3 = self._rho_1(features2)
        features4 = self._rho_2(features3)
        features5 = self._rho_3(features4)

        q = self._out(features5)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


class Q4(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q4, self).__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features
        self._h1 = nn.Linear(n_input, n_features)

        self._rho_0 = ResidualBlock(n_features)
        self._rho_1 = ResidualBlock(n_features)
        self._rho_2 = ResidualBlock(n_features)
        self._rho_3 = ResidualBlock(n_features)
        self._rho_4 = ResidualBlock(n_features)
        self._out = LinearOutput(n_features, self.n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self,state, action=None):

        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._rho_0(features1)
        features3 = self._rho_1(features2)
        features4 = self._rho_2(features3)
        features5 = self._rho_3(features4)
        features6 = self._rho_4(features5)

        q = self._out(features6)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted


class Q5(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q5, self).__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features
        self._h1 = nn.Linear(n_input, n_features)

        self._rho_0 = ResidualBlock(n_features)
        self._rho_1 = ResidualBlock(n_features)
        self._rho_2 = ResidualBlock(n_features)
        self._rho_3 = ResidualBlock(n_features)
        self._rho_4 = ResidualBlock(n_features)
        self._rho_5 = ResidualBlock(n_features)
        self._out = LinearOutput(n_features, self.n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self,state, action=None):

        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._rho_0(features1)
        features3 = self._rho_1(features2)
        features4 = self._rho_2(features3)
        features5 = self._rho_3(features4)
        features6 = self._rho_4(features5)
        features7 = self._rho_5(features6)

        q = self._out(features7)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted