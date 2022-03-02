import math
from textwrap import wrap
import numpy as np
import torch
import torch.nn.functional as F


class AttentionField(object):
    """ Attention mechanism operations for the intercommunications between neurons. """

    def __init__(self, k_dim, v_dim):
        """ Initialization

        Args:
            k_dim (int): dimension of keys and queries. 
            v_dim (int): dimension of value. """

        self.k_dim = k_dim
        self.v_dim = v_dim
        self.self_attention = False
        self.reset_entries()

    def add_entries(self, queries, keys, values):
        """ Define the values of the three components in the attention mechanism. 

        Args:
            queries (np.ndarrays)
            keys (np.ndarrays)
            values (np.ndarrays)"""

        if type(queries) is np.ndarray:
            if queries.shape[1] == self.k_dim:
                self.queries = torch.cat(
                    (self.queries, torch.tensor(queries)), 0)
            else:
                print("Attention field: queries do not have correct shape. Expected {} and got {}".format(
                    self.k_dim, queries.shape[1]))
        if type(keys) is np.ndarray and type(values) is np.ndarray:
            if keys.shape[1] == self.k_dim and values.shape[1] == self.v_dim:
                self.keys = torch.cat((self.keys, torch.tensor(keys)), 0)
                self.values = torch.cat((self.values, torch.tensor(values)), 0)
            else:
                print("Attention field: keys or values do not have correct shape. Expected {},{} and got {},{}".format(
                    self.k_dim, self.v_dim, keys.shape[1], values.shape[1]))

    def scaled_dot_product(self, q, k, v, stage):
        """ Math operation required to calculate the allignment of keys and queries.

        Args:
            q (int)
            k (int)
            v (int)
            mask (bool): Wether to use a mask in a portion of the result. Not used now. """

        n_q = q.size()[0]
        n_k = k.size()[0]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(q.size()[-1])
        if not self.self_attention and stage == 1 and n_k > n_q:
            diagonal_mask = np.diag(-9999*np.ones(n_q), n_k-n_q)[:n_q]
            attn_logits = attn_logits + diagonal_mask

        self.set_attention = F.softmax(attn_logits, dim=-1)
        self.avg_values = torch.matmul(self.set_attention, v)

    def reset_entries(self):
        """ Clean values. """

        self.queries = torch.tensor(np.array([]))
        self.keys = torch.tensor(np.array([]))
        self.values = torch.tensor(np.array([]))

    def run_step(self, stage):
        """ Calculate the scale dot product and reset attention players.

        Returns: 
            avg_values (np.array): Weighted sum of values of attended enurons.
            set_attention (np.array): Attention weights computed for each key neuron. """

        self.scaled_dot_product(self.queries, self.keys, self.values, stage)
        self.reset_entries()
        return self.avg_values.numpy(), self.set_attention.numpy()
