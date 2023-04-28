import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r = self.r_act(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh) # TODO
        self.z = self.z_act(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh) # TODO
        self.n = self.h_act(self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh)) # TODO
        h_t = (1 - self.z) * self.n + self.z * h_prev_t # TODO
        

        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        x = np.reshape(self.x, (1, -1))
        hidden = np.reshape(self.hidden, (1, -1))
        
        # 2) Transpose all calculated dWs...
        dn = delta*(1-self.z)*self.h_act.backward()
        dz = delta*(self.hidden-self.n)*self.z_act.backward()
        dr = dn*(self.Wnh @ self.hidden+self.bnh)*self.r_act.backward()
    
        # 3) Compute all of the derivatives
        self.dWrx += dr.T @ x
        self.dWzx += dz.T @ x
        self.dWnx += dn.T @ x
        self.dWrh += dr.T @ hidden
        self.dWzh += dz.T @ hidden
        self.dWnh += (dn*self.r).T @ hidden
        self.dbrx += np.sum(dr, axis=0)
        self.dbzx += np.sum(dz, axis=0)
        self.dbnx += np.sum(dn, axis=0)
        self.dbrh += np.sum(dr, axis=0)
        self.dbzh += np.sum(dz, axis=0)
        self.dbnh += np.sum(dn*self.r, axis=0)
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.
        dx = dn @ self.Wnx + dz @ self.Wzx + dr @ self.Wrx
        dh_prev_t = self.r*dn @ self.Wnh + dz @ self.Wzh + dr @ self.Wrh + self.z*delta

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        
        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t