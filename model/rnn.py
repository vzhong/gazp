from torch import nn


class LockedDropout(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = m.div_(1 - dropout)
        mask.requires_grad = False
        mask = mask.expand_as(x)
        return mask * x


def run_rnn(rnn, inp, lens, state0=None):
    pack = nn.utils.rnn.pack_padded_sequence(inp, lens.cpu(), batch_first=True, enforce_sorted=False)
    out, state = rnn.forward(pack, state0)
    unpack, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=0, total_length=inp.size(1))
    return unpack, state
