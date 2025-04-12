import torch
import matplotlib.pyplot as plt


class InceptionBlock(torch.nn.Module):
    """
    Inception block to handle n_modalities at onces using 
    additive, multiplicative, and nonlinear fusion.
    All resulting latent_representations are combined
    using weighted sum based on trainable coefficients.
    """

    def __init__(self, size, n_modalities=2, nonlinear=True):
        """Initialize inception block module.

        :param size: Size of all modalities.
        :param n_modalities: Number of modalities, defaults to 2.
        :param nonlinear: Enable nonlinear fusion of modalities, defaults to True.
        """
        super().__init__()
        self.size = size
        self.n_modalities = n_modalities
        self.nonlinear=nonlinear

        if nonlinear:
            self.func = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(n_modalities * size, size),
                torch.nn.ReLU(),
                torch.nn.Linear(size, size)
            )
            self.param_concat = torch.nn.Parameter(torch.rand((1,)))
        else:
            self.func = lambda _: 0.0
        self.param_add = torch.nn.Parameter(torch.rand((1,)))
        self.param_mul = torch.nn.Parameter(torch.rand((1,)))

    def get_coefs(self):
        """ Get coefficients based on softmax. """
        if self.nonlinear:
            exp_concat = torch.exp(self.param_concat)
            exp_add = torch.exp(self.param_add)
            exp_mul = torch.exp(self.param_mul)
            exp_sum = exp_concat + exp_add + exp_mul
            coef_concat = exp_concat / exp_sum
            coef_add = exp_add / exp_sum
            coef_mul = exp_mul / exp_sum
            return coef_concat, coef_add, coef_mul
        else:
            exp_add = torch.exp(self.param_add)
            exp_mul = torch.exp(self.param_mul)
            exp_sum = exp_add + exp_mul
            coef_add = exp_add / exp_sum
            coef_mul = exp_mul / exp_sum
            return 0.0, coef_add, coef_mul

    def forward(self, modalities):
        coef_concat, coef_add, coef_mul = self.get_coefs()
        output = coef_concat * self.func(modalities)
        output += coef_add * torch.sum(modalities, axis=1)
        output += coef_mul * torch.prod(modalities, axis=1)
        return output

    def debug(self, filename):
        coef_concat, coef_add, coef_mul = self.get_coefs()
        coef_concat = coef_concat.item()
        coef_add = coef_add.item()
        coef_mul = coef_mul.item()
        plt.bar(['coef_concat', 'coef_add', 'coef_mul'],
                [coef_concat, coef_add, coef_mul])
        plt.savefig(filename)
        plt.close()
