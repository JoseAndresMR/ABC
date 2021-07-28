import math
import numpy as np
import torch
import torch.nn.functional as F

class AttentionField(object):

    def __init__(self, k_dim, v_dim):
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.resetEntries()

    def scaledDotProduct(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        self.set_attention = F.softmax(attn_logits, dim=-1)
        self.avg_values = torch.matmul(self.set_attention, v)
        
    def addEntries(self, queries, keys, values):
        if type(queries) is np.ndarray:
            if queries.shape[1] == self.k_dim:
                self.queries = torch.cat((self.queries, torch.tensor(queries)), 0)
            else:
                print("Attention field: queries do not have correct shape. Expected {} and got {}".format(self.k_dim, queries.shape[1]))
        if type(keys) is np.ndarray and type(values) is np.ndarray:
            if keys.shape[1] == self.k_dim and values.shape[1] == self.v_dim:
                self.keys = torch.cat((self.keys, torch.tensor(keys)), 0)
                self.values = torch.cat((self.values, torch.tensor(values)), 0)
            else:
                print("Attention field: keys or values do not have correct shape. Expected {},{} and got {},{}".format(self.k_dim, self.v_dim, keys.shape[1], values.shape[1]))

    def resetEntries(self):
        self.queries = torch.tensor(np.array([]))
        self.keys = torch.tensor(np.array([]))
        self.values = torch.tensor(np.array([]))

    def runStep(self):
        # print("Attention field: running step with {} queries, {} keys and {} values".format(self.queries.shape, self.keys.shape, self.values.shape))
        self.scaledDotProduct(self.queries, self.keys, self.values)
        self.resetEntries()
        return self.avg_values.numpy(), self.set_attention.numpy()

    def test(self):
        q = torch.randn(3, 2)
        k = torch.randn(3, 2)
        v = torch.randn(3, 10)
        self.addEntries(q, k, v)
        q = torch.randn(1, 2)
        k = torch.randn(5, 2)
        v = torch.randn(5, 10)
        self.addEntries(q, k, v) 
        values, attention = self.runStep()
        print("Q\n", q)
        print("K\n", k)
        print("V\n", v)
        print("Values\n", values)
        print("Attention\n", attention)

# af = AttentionField(2,10)
# af.test()