import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dNormalGamma(nn.Module):
    def __init__(self):
        super(Conv2dNormalGamma, self).__init__()
        self.Conv_2304 = nn.Conv2d(2304, 4 * 2304)
        self.Conv_1152 = nn.Conv2d(1152, 4 * 1152)
        self.Conv_576 = nn.Conv2d(576, 4 * 576)
        self.Conv_288 = nn.Conv2d(288, 4 * 288)
        self.Conv_144 = nn.Conv2d(144, 4 * 144)
        self.Conv_72 = nn.Conv2d(72, 4 * 72)


    def evidence(self, x):
        return F.softplus(x)
    
        # def changeDim(self,x):
        #     self.in_dim = int(x.size()[-1])
        #     self.out_dim = int(x.size()[-1])
        #     self.Conv = nn.Linear(self.in_dim * 2, 4 * self.out_dim).cuda()
        #
        # def getDim(self):
        #     print(self.in_dim,self.out_dim)
    
    
    def forward(self, x):
        if int(x.size()[-1]) == 2304:
            output = self.Conv_2304(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 1152:
            output = self.Conv_1152(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 576:
            output = self.Conv_576(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 288:
            output = self.Conv_288(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
    
        elif int(x.size()[-1]) == 144:
            output = self.Conv_144(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 72:
            output = self.Conv_72(x)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
    


