# coding: utf-8

import torch


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999, start_epoch=10):
        # save arguments
        self.model = model
        self.decay = decay
        self.start_epoch = start_epoch

        # register exponential moving averages of parameters
        self.ema = {}
        for name, param in model.named_parameters():
            self.ema[name] = param.clone()

    def swap(self):
        """swaps model parameters with the stored averages"""
        for name, param in self.model.named_parameters():
            tmp = param.clone()
            param.data = self.ema[name]
            self.ema[name] = tmp

    def overwrite(self):
        """overwrites model parameters with the stored averages (no backup copy is kept)"""
        for name, param in self.model.named_parameters():
            param.data = self.ema[name]

    def __call__(self, epoch):
        """
        Applies exponential moving average to parameters
        (should be called after every training step)
        """

        # this allows to only start after a certain number of epochs with the
        # averaging => in the first few epochs, parameters change a lot and
        # it is better to start averaging a little bit later
        if epoch > self.start_epoch:
            alpha = 1 - self.decay
        else:
            alpha = 1

        # apply averaging
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    self.ema[name].data -= alpha * (self.ema[name] - param.data)
