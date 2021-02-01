from torch import nn
from torch.nn import functional as F


class DotAttn(nn.Module):

    def __init__(self, dropout=0, enc_trans=None, dec_trans=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dec_trans = dec_trans or nn.Identity()
        self.enc_trans = enc_trans or nn.Identity()

    def forward(self, enc, enc_mask, dec):
        enc_trans = self.dropout(self.enc_trans(enc))
        dec_trans = self.dropout(self.dec_trans(dec))
        Bdec, Tdec, Ddec = dec_trans.size()
        Benc, Tenc, Denc = enc_trans.size()
        dec_trans_exp = dec_trans.unsqueeze(1).expand(Bdec, Tenc, Tdec, Ddec)
        enc_trans_exp = enc_trans.unsqueeze(2).expand(Benc, Tenc, Tdec, Denc)
        mask_exp = enc_mask.unsqueeze(2).expand(Benc, Tenc, Tdec)
        raw_scores_exp = dec_trans_exp.mul(enc_trans_exp).sum(3)
        raw_scores_exp -= (1-mask_exp) * 1e20
        scores_exp = F.softmax(raw_scores_exp, dim=1)
        context_exp = scores_exp.unsqueeze(3).expand_as(enc_trans_exp).mul(enc_trans_exp).sum(1)
        return context_exp, scores_exp


class SelfAttn(nn.Module):

    def __init__(self, denc, dropout=0, enc_trans=None, dec_trans=None):
        super().__init__()
        self.scorer = nn.Linear(denc, 1)
        self.dropout = nn.Dropout(dropout)
        self.enc_trans = enc_trans or nn.Identity()

    def forward(self, enc, enc_mask):
        enc_trans = self.dropout(self.enc_trans(enc))
        raw_scores = self.scorer(enc).squeeze(2)
        scores = F.softmax(raw_scores - (1-enc_mask) * 1e20, dim=1)
        context = scores.unsqueeze(2).expand_as(enc_trans).sum(1)
        return context, scores
