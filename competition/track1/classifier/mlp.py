import torch.nn as nn

class MLP(nn.Module):

    """Convolutional encoder for image-based observations."""
    def __init__(self, input_dim, output_dim, hidden_size=64, hidden_depth=2, activ=nn.Tanh, output_mod=None, init_func=None):
        super().__init__()

        if init_func is None:
            init_func = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))
        if hidden_depth == 0:
            mods = [init_func(nn.Linear(input_dim, output_dim))]
        else:
            mods = [init_func(nn.Linear(input_dim, hidden_size)), activ()]
            for i in range(hidden_depth - 1):
                mods += [init_func(nn.Linear(hidden_size, hidden_size)), activ()]
            mods.append(init_func(nn.Linear(hidden_size, output_dim)))
        if output_mod is not None:
            mods.append(output_mod)
        self.main = nn.Sequential(*mods)

    def forward(self, obs, detach=False):
        out = self.main(obs)
        if detach:
            out = out.detach()
        return out