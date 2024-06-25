from torch.utils.data.distributed import DistributedSampler


def distributed_sampler(dataset):
    return DistributedSampler(dataset)
