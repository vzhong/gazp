import copy
import torch
from torch import nn, distributions
from torch.nn import functional as F
from model import attention


class Beam:

    def __init__(self, eos_ind=None):
        self.scores = []
        self.inds = []
        self.state = None
        self.eos_ind = eos_ind

    def add(self, score, ind, state):
        self.scores.append(score)
        self.inds.append(ind)
        self.state = state

    def clone(self):
        return copy.deepcopy(self)

    @property
    def beam_score(self):
        score = total = 0
        for i, s in zip(self.inds, self.scores):
            score += s
            total += 1
            if i == self.eos_ind:
                break
        return score / total if total else 0

    def terminated(self):
        return self.eos_ind in self.inds

    def __repr__(self):
        return 'Beam({}, {})'.format(self.beam_score, self.inds)

    def __eq__(self, other):
        return self.beam_score == other.beam_score

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.beam_score < other.beam_score

    def __hash__(self):
        return hash(str(self.inds))


class PointerDecoder(nn.Module):

    def __init__(self, demb, denc, ddec, dropout=0, num_layers=1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(demb, ddec, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.attn = attention.DotAttn(dropout=dropout, dec_trans=nn.Linear(ddec, denc))
        self.go = nn.Parameter(torch.Tensor(demb))
        self.feat_lin = nn.Linear(denc+ddec, demb+1)
        nn.init.uniform_(self.go, -0.1, 0.1)

    def forward(self, emb, emb_mask, enc, enc_mask, state0=None, gt=None, max_len=50, batch=None):
        B, n_candidates, _ = emb.size()
        emb_t = self.go.unsqueeze(0).repeat(B, 1)
        picked_t = torch.zeros(B, n_candidates).to(emb.device)
        state_t = state0
        scores = []
        debug = []
        for t in range(gt.size(1) if gt is not None else max_len):
            emb_t = self.dropout(emb_t)
            h_t, state_t = self.rnn(emb_t.unsqueeze(1), state_t)
            h_t = self.dropout(h_t)
            attn_t, _ = self.attn(enc, enc_mask, h_t)
            attn_t = self.dropout(attn_t)
            feat_t = self.feat_lin(torch.cat([h_t, attn_t], dim=2))

            cand_t = torch.cat([emb, picked_t.unsqueeze(2)], dim=2)
            score_t = feat_t.expand_as(cand_t).mul(cand_t).sum(2)
            score_t = score_t - (1-emb_mask)*1e20
            max_t = gt[:, t] if gt is not None else score_t.max(1)[1]
            for i, (p, c) in enumerate(zip(picked_t, max_t)):
                p[c].fill_(1)
            emb_t = torch.stack([emb[i, j] for i, j in enumerate(max_t.tolist())], dim=0)
            debug.append((max_t[0], emb_t[0]))
            scores.append(score_t)
        return torch.stack(scores, dim=1)

    def sample(self, emb, emb_mask, enc, enc_mask, eos_ind, state_0=None, max_len=50, batch=None):
        B, n_candidates, _ = emb.size()
        emb_t = self.go.unsqueeze(0).repeat(B, 1)
        picked_t = torch.zeros(B, n_candidates).to(emb.device)
        state_t = state_0
        samples = []
        for t in range(max_len):
            emb_t = self.dropout(emb_t)
            h_t, state_t = self.rnn(emb_t.unsqueeze(1), state_t)
            h_t = self.dropout(h_t)
            attn_t, _ = self.attn(enc, enc_mask, h_t)
            attn_t = self.dropout(attn_t)
            feat_t = self.feat_lin(torch.cat([h_t, attn_t], dim=2))

            cand_t = torch.cat([emb, picked_t.unsqueeze(2)], dim=2)
            score_t = feat_t.expand_as(cand_t).mul(cand_t).sum(2)
            score_t = score_t - (1-emb_mask)*1e20
            norm_t = F.softmax(score_t, dim=1)
            c = distributions.Categorical(norm_t)
            max_t = c.sample()
            for i, (p, c) in enumerate(zip(picked_t, max_t)):
                p[c].fill_(1)
            emb_t = torch.stack([emb[i, j] for i, j in enumerate(max_t.tolist())], dim=0)
            samples.append(max_t)
        samples = torch.stack(samples, dim=1).tolist()
        proc = []
        for s in samples:
            if eos_ind in s:
                s = s[:s.index(eos_ind)]
            proc.append(s)
        return proc

    def beam_search_step(self, emb_t, picked_t, emb, emb_mask, enc, enc_mask, state_t, beam_size=3):
        emb_t = self.dropout(emb_t)
        h_t, state_t = self.rnn(emb_t.unsqueeze(1), state_t)
        h_t = self.dropout(h_t)
        attn_t, _ = self.attn(enc, enc_mask, h_t)
        attn_t = self.dropout(attn_t)
        feat_t = self.feat_lin(torch.cat([h_t, attn_t], dim=2))

        cand_t = torch.cat([emb, picked_t.unsqueeze(2)], dim=2)
        score_t = feat_t.expand_as(cand_t).mul(cand_t).sum(2)
        score_t = score_t - (1-emb_mask)*1e20
        norm_score_t = torch.log_softmax(score_t, dim=1)
        return torch.topk(norm_score_t, beam_size, dim=1), score_t, state_t

    def beam_search(self, emb, emb_mask, enc, enc_mask, eos_ind, max_len=50, batch=None, beam_size=3):
        old = self.forward(emb, emb_mask, enc, enc_mask)
        device = emb.device
        B, n_candidates, _ = emb.size()
        out = []
        for i, (embi, emb_maski, enci, enc_maski, ex) in enumerate(zip(emb, emb_mask, enc, enc_mask, batch)):
            embi = embi.unsqueeze(0).repeat(beam_size, 1, 1)
            emb_maski = emb_maski.unsqueeze(0).repeat(beam_size, 1)
            enci = enci.unsqueeze(0).repeat(beam_size, 1, 1)
            enc_maski = enc_maski.unsqueeze(0).repeat(beam_size, 1)

            emb_t = self.go.unsqueeze(0).repeat(beam_size, 1)
            picked_t = torch.zeros(beam_size, n_candidates).to(emb.device)
            state_t = None
            beams = [Beam(eos_ind=eos_ind) for _ in range(beam_size)]
            for t in range(max_len):
                (top_scores, top_inds), scores, (h_t, c_t) = self.beam_search_step(emb_t, picked_t, embi, emb_maski, enci, enc_maski, state_t, beam_size)
                new_beams = set()
                for tsi, tii, hi, ci, bi in zip(top_scores.tolist(), top_inds.tolist(), h_t.transpose(1, 0), c_t.transpose(1, 0), beams):
                    if bi.terminated():
                        new_beams.add(bi)
                    else:
                       for tsij, tiij in zip(tsi, tii):
                           b = bi.clone()
                           b.add(score=tsij, ind=tiij, state=(hi, ci))
                           new_beams.add(b)
                beams = sorted(list(new_beams), reverse=True)[:beam_size]

                if all([b.terminated() for b in beams]):
                    break

                h_t, c_t, emb_t, picked_t = [], [], [], []
                zeros = torch.zeros(n_candidates)
                for b, e in zip(beams, embi):
                    hb, cb = b.state
                    h_t.append(hb)
                    c_t.append(cb)
                    emb_t.append(e[b.inds[-1]])
                    picked_t.append(torch.tensor([x in b.inds for x in range(n_candidates)], dtype=torch.float))

                h_t = torch.stack(h_t, dim=1)
                c_t = torch.stack(c_t, dim=1)
                state_t = h_t, c_t
                emb_t = torch.stack(emb_t, dim=0).to(device)
                picked_t = torch.stack(picked_t, dim=0).to(device)
            out.append(beams)
        return out
