# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class AdamW():
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay=weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]
        
        self.W = [i.W for i in self.l]
        self.b = [i.b for i in self.l]

    def step(self):
        self.t += 1
        if self.t==1:
            for layer_id, layer in enumerate(self.l):
                self.l[layer_id].W -= self.l[layer_id].W * self.lr * self.weight_decay
                self.l[layer_id].b -= self.l[layer_id].b * self.lr*self.weight_decay
                gW = layer.dLdW
                gb = layer.dLdb
                # TODO: Calculate updates for weight
                self.m_W[layer_id] = self.beta1 * self.m_W[layer_id]+(1.0-self.beta1)*gW
                self.v_W[layer_id] = self.beta2 * self.v_W[layer_id]+(1.0-self.beta2)*(gW**2)
                self.m_W[layer_id] = self.m_W[layer_id] / (1.0-self.beta1**self.t)
                self.v_W[layer_id] = self.v_W[layer_id] / (1.0-self.beta2**self.t)
                # TODO: calculate updates for bias
                self.m_b[layer_id] = self.beta1 * self.m_b[layer_id]+(1.0-self.beta1)*gb
                self.v_b[layer_id] = self.beta2 * self.v_b[layer_id]+(1.0-self.beta2)*(gb**2)
                self.m_b[layer_id] = self.m_b[layer_id] / (1.0-self.beta1**self.t)
                self.v_b[layer_id] = self.v_b[layer_id] / (1.0-self.beta2**self.t)
                # TODO: Perform weight and bias updates
                self.l[layer_id].W -= (self.lr*self.m_W[layer_id]) / ((self.v_W[layer_id]+self.eps)**0.5)
                self.l[layer_id].b -= (self.lr*self.m_b[layer_id]) / ((self.v_b[layer_id]+self.eps)**0.5)
                self.W[layer_id] = self.l[layer_id].W
                self.b[layer_id] = self.l[layer_id].b
        else:
            for layer_id, layer in enumerate(self.l):
                self.l[layer_id].W = self.W[layer_id]
                self.l[layer_id].b = self.b[layer_id]
        return None
