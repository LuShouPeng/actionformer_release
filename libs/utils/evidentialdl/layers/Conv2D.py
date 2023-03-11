import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dNormalGamma(nn.Module):
    def __init__(self):
        super(Conv2dNormalGamma, self).__init__()
        self.Conv_2304 = nn.Conv1d(2304, 4 * 2304,kernel_size=1)
        self.Conv_1152 = nn.Conv1d(1152, 4 * 1152,kernel_size=1)
        self.Conv_576 = nn.Conv1d(576, 4 * 576,kernel_size=1)
        self.Conv_288 = nn.Conv1d(288, 4 * 288,kernel_size=1)
        self.Conv_144 = nn.Conv1d(144, 4 * 144,kernel_size=1)
        self.Conv_72 = nn.Conv1d(72, 4 * 72,kernel_size=1)


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
            x = x.permute(0, 2, 1)
            output = self.Conv_2304(x).permute(0,2,1)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 1152:
            x = x.permute(0, 2, 1)
            output = self.Conv_1152(x).permute(0,2,1)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 576:
            x = x.permute(0, 2, 1)
            output = self.Conv_576(x).permute(0,2,1)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 288:
            x = x.permute(0, 2, 1)
            output = self.Conv_288(x).permute(0,2,1)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
    
        elif int(x.size()[-1]) == 144:
            x = x.permute(0, 2, 1)
            output = self.Conv_144(x).permute(0,2,1)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
        elif int(x.size()[-1]) == 72:
            x = x.permute(0, 2, 1)
            output = self.Conv_72(x).permute(0,2,1)
            mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)
            v = self.evidence(logv)
            alpha = self.evidence(logalpha) + 1
            beta = self.evidence(logbeta)
    
            aleatoric = beta / (alpha - 1)
            epistemic = beta / v * (alpha - 1)
    
            return mu, v, alpha, beta, aleatoric, epistemic
    


