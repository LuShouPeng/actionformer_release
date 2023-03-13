import torch.nn as nn
import torch.nn.functional as F
import torch

class DenseNormal(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseNormal, self).__init__()
        self.in_dim = int(in_dim)
        self.out_features = int(out_dim)
        self.dense = nn.Linear(self.in_dim, 2 * self.out_features)

    def forward(self, x):
        output = self.dense(x)
        mu, logsigma = output.chunk(2, dim=-1)
        sigma = F.softplus(logsigma) + 1e-6
        return torch.cat([mu, sigma], dim=-1)


class DenseNormalGamma(nn.Module):
    # def __init__(self, in_dim, out_dim):
    def __init__(self):
        # super(DenseNormalGamma, self).__init__()
        # self.in_dim = int(in_dim)
        # self.out_dim = int(out_dim)
        # self.dense = nn.Linear(self.in_dim, 4 * self.out_dim)
        super(DenseNormalGamma, self).__init__()
        # self.in_dim = int(in_dim)
        # self.out_dim = int(out_dim)
        self.dense_2304 = nn.Linear(2304, 4 * 2304)
        self.dense_1152 = nn.Linear(1152, 4 * 1152)
        self.dense_576 = nn.Linear(576, 4 * 576)
        self.dense_288 = nn.Linear(288, 4 * 288)
        self.dense_144 = nn.Linear(144, 4 * 144)
        self.dense_72 = nn.Linear(72, 4 * 72)

    def evidence(self, x):

        return F.softplus(x)

    # def changeDim(self,x):
    #     self.in_dim = int(x.size()[-1])
    #     self.out_dim = int(x.size()[-1])
    #     self.dense = nn.Linear(self.in_dim * 2, 4 * self.out_dim).cuda()
    #
    # def getDim(self):
    #     print(self.in_dim,self.out_dim)

    def forward(self, x):
        if int(x.size()[-1]) == 2304:
            output = self.dense_2304(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)

            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)

            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 1152:
            output = self.dense_1152(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)

            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)

            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 576:
            output = self.dense_576(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)

            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)

            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 288:
            output = self.dense_288(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)

            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)

            return mu, v, alpha, beta, aleatoric, epistemic
        
        elif int(x.size()[-1]) == 144:
            output = self.dense_144(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)

            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)

            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 72:
            output = self.dense_72(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)

            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)

            return mu, v, alpha, beta, aleatoric, epistemic


class Conv2dNormalGamma:
    pass