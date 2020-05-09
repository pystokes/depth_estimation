#!/usr/bin/python3
# -*- coding: utf-8 -*-

from torch import optim

class Optimizers(object):

    @classmethod
    def get_optimizer(self, config, params):

        if config.type == 'sgd':
            return optim.SGD(params=params,
                             lr=config.sgd.lr,
                             momentum=config.sgd.momentum,
                             weight_decay=config.sgd.weight_decay)

        else:
            """
            Default optimizer is Adam
            """
            return optim.Adam(params=params,
                              lr=config.adam.lr,
                              betas=config.adam.betas,
                              eps=config.adam.eps,
                              weight_decay=config.adam.weight_decay,
                              amsgrad=config.adam.amsgrad)
        

if __name__ == '__main__':
    pass
