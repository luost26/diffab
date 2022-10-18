import numpy as np
import torch
from easydict import EasyDict

from .misc import BlackHole


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2, )
        )
    else:
        raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    elif cfg.type is None:
        return BlackHole()
    else:
        raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def get_warmup_sched(cfg, optimizer):
    if cfg is None: return BlackHole()
    lambdas = [lambda it : (it / cfg.max_iters) if it <= cfg.max_iters else 1 for _ in optimizer.param_groups]
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return warmup_sched


def log_losses(out, it, tag, logger=BlackHole(), writer=BlackHole(), others={}):
    logstr = '[%s] Iter %05d' % (tag, it)
    logstr += ' | loss %.4f' % out['overall'].item()
    for k, v in out.items():
        if k == 'overall': continue
        logstr += ' | loss(%s) %.4f' % (k, v.item())
    for k, v in others.items():
       logstr += ' | %s %2.4f' % (k, v)
    logger.info(logstr)

    for k, v in out.items():
        if k == 'overall':
            writer.add_scalar('%s/loss' % tag, v, it)
        else:
            writer.add_scalar('%s/loss_%s' % (tag, k), v, it)
    for k, v in others.items():
        writer.add_scalar('%s/%s' % (tag, k), v, it)
    writer.flush()


class ValidationLossTape(object):

    def __init__(self):
        super().__init__()
        self.accumulate = {}
        self.others = {}
        self.total = 0

    def update(self, out, n, others={}):
        self.total += n
        for k, v in out.items():
            if k not in self.accumulate:
                self.accumulate[k] = v.clone().detach()
            else:
                self.accumulate[k] += v.clone().detach()

        for k, v in others.items():
            if k not in self.others:
                self.others[k] = v.clone().detach()
            else:
                self.others[k] += v.clone().detach()
        

    def log(self, it, logger=BlackHole(), writer=BlackHole(), tag='val'):
        avg = EasyDict({k:v / self.total for k, v in self.accumulate.items()})
        avg_others = EasyDict({k:v / self.total for k, v in self.others.items()})
        log_losses(avg, it, tag, logger, writer, others=avg_others)
        return avg['overall']


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        if device == 'cpu':
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def reweight_loss_by_sequence_length(length, max_length, mode='sqrt'):
    if mode == 'sqrt':
        w = np.sqrt(length / max_length)
    elif mode == 'linear':
        w = length / max_length
    elif mode is None:
        w = 1.0
    else:
        raise ValueError('Unknown reweighting mode: %s' % mode)
    return w


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
