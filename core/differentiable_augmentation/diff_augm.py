from .AugmentPipe_kornia import AugmentPipe_kornia
from .augment import AugmentPipe


class augment_pipe():
    def __init__(self, opt):
        if opt.use_kornia_augm:
            self.augment_func = AugmentPipe_kornia(
                opt.prob_augm, opt.no_masks).to(opt.device)
        else:
            self.augment_func = AugmentPipe().to(opt.device)

    def __call__(self, batch, real=True):
        return self.augment_func(batch)
