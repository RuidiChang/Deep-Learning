# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # TODO: Generate mask and apply to x
            mask=[]
            for i in range(x.shape[0]):
                partial = np.random.binomial(1, self.p, x.shape[1])
                where_0 = np.where(partial == 0)
                where_1 = np.where(partial == 1)
                partial[where_0] = 1
                partial[where_1] = 0
                self.partial=partial
                mask.append(self.partial)
                x[i]=x[i]*self.partial
            self.mask = np.reshape(mask, x.shape)
            return x/(1.0-self.p)
        else:
            # TODO: Return x as is
            return x
        
    def backward(self, delta):
        # TODO: Multiply mask with delta and return
        output=delta*self.mask
        return output