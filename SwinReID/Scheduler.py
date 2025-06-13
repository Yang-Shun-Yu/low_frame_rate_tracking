WARMUP_EPOCH_NET = 10
WARMUP_EPOCH_CENTER = 5
DECAY_FACTOR_NET = 0.1
DECAY_FACTOR_CENTER = 0.1
from torch.optim.lr_scheduler import LambdaLR

def warmup_schedule_net(epoch):
    if epoch < WARMUP_EPOCH_NET:
        return epoch / WARMUP_EPOCH_NET
    else:
        return 1.0

def decay_schedule_net(epoch):
    if epoch < WARMUP_EPOCH_NET:
        return 1.0
    elif WARMUP_EPOCH_NET <= epoch <= 25:
        return DECAY_FACTOR_NET
    else:
        return DECAY_FACTOR_NET ** 2

def combined_schedule_net(epoch):
    return warmup_schedule_net(epoch) * decay_schedule_net(epoch)

def warmup_schedule_center(epoch):
    if epoch < WARMUP_EPOCH_CENTER:
        return epoch / WARMUP_EPOCH_CENTER
    else:
        return 1.0

def decay_schedule_center(epoch):
    if epoch < WARMUP_EPOCH_CENTER:
        return 1.0
    elif WARMUP_EPOCH_CENTER <= epoch <= 20:
        return DECAY_FACTOR_CENTER
    else:
        return DECAY_FACTOR_CENTER ** 2

def combined_schedule_center(epoch):
    return warmup_schedule_center(epoch) * decay_schedule_center(epoch)


def get_scheduler_net_center(optimizer):
    return LambdaLR(optimizer, lr_lambda=[combined_schedule_net, combined_schedule_center])

def get_scheduler_net(optimizer):
    return LambdaLR(optimizer, lr_lambda=combined_schedule_net)