import logging
import tqdm, torch, torchvision
import numpy as np


class TqdmToLogger(tqdm):
    def __init__(self, *args, logger=None,
                 mininterval=0.1,
                 bar_format='{desc:<}{percentage:3.0f}% |{bar:20}| [{n_fmt:6s}/{total_fmt}]',
                 desc=None,
                 **kwargs
                 ):
        self._logger = logger
        super().__init__(*args, mininterval=mininterval,
                         bar_format=bar_format, desc=desc, **kwargs)

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logger

    def display(self, msg=None, pos=None):
        if not self.n:
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg.strip('\r\n\t '))


logger = logging.getLogger(__name__)


def simulate_split(dataset):
    """Split data indices using labels.

    Args:
        args (argparser): arguments
        dataset (dataset): raw dataset instance to be split

    Returns:
        split_map (dict): dictionary with key is a client index and a corresponding value is a list of indices

    '--K', help='number of total cilents participating in federated training', type=int, default=100
    '--test_size', help='a fraction of local hold-out dataset for evaluation (-1 for assigning pre-defined test split as local holdout set)', type=float, choices=[Range(-1, 1.)], default=0.2
    '--cncntrtn', help='a concentration parameter for Dirichlet distribution (valid only if `split_type` is `diri`)', type=float, default=0.1
    '--mincls', help='the minimum number of distinct classes per client (valid only if `split_type` is `patho` or `diri`)', type=int, default=2
    """
    K = 100
    test_size = 0.2
    num_classes = 10
    cncntrtn = 0.1
    mincls = 2
    # Non-IID split proposed in (Hsu et al., 2019); simulation of non-IID split scenario using Dirichlet distribution
    MIN_SAMPLES = int(1 / test_size)

    total_counts = len(dataset.targets)
    _, unique_inverse, unique_counts = np.unique(
        dataset.targets, return_inverse=True, return_counts=True)
    class_indices = np.split(np.argsort(
        unique_inverse), np.cumsum(unique_counts[:-1]))

    # calculate ideal samples counts per client
    ideal_counts = len(dataset.targets) // K
    if ideal_counts < 1:
        err = f'[SIMULATE] Decrease the number of participating clients (`args.K` < {K})!'
        logger.exception(err)
        raise Exception(err)

    # split dataset
    # define temporary container
    assigned_indices = []

    # NOTE: it is possible that not all samples be consumed, as it is intended for satisfying each clients having at least `MIN_SAMPLES` samples per class
    for k in TqdmToLogger(range(K), logger=logger, desc='[SIMULATE] ...assigning to clients... '):
        # for current client of which index is `k`
        curr_indices = []
        satisfied_counts = 0

        # ...until the number of samples close to ideal counts is filled
        while satisfied_counts < ideal_counts:
            # define Dirichlet distribution of which prior distribution is an uniform distribution
            diri_prior = np.random.uniform(size=num_classes)

            # sample a parameter corresponded to that of categorical distribution
            cat_param = np.random.dirichlet(
                alpha=cncntrtn * diri_prior)

            # try to sample by amount of `ideal_counts``
            sampled = np.random.choice(
                num_classes, ideal_counts, p=cat_param)

            # count per-class samples
            unique, counts = np.unique(sampled, return_counts=True)
            if len(unique) < mincls:
                continue

            # filter out sampled classes not having as much as `MIN_SAMPLES`
            required_counts = counts * (counts > MIN_SAMPLES)

            # assign from population indices split by classes
            for idx, required_class in enumerate(unique):
                if required_counts[idx] == 0:
                    continue
                sampled_indices = class_indices[required_class][:required_counts[idx]]
                curr_indices.append(sampled_indices)
                class_indices[required_class] = class_indices[required_class][:required_counts[idx]]
            satisfied_counts += sum(required_counts)

        # when enough samples are collected, go to next clients!
        assigned_indices.append(np.concatenate(curr_indices))

    # construct a hashmap
    split_map = {k: assigned_indices[k] for k in range(K)}
    return split_map

dataset = torchvision.datasets.MNIST(root='./data',
                                     train=True,
                                     transform=torchvision.transforms.ToTensor(),
                                     download=True)

split_data = simulate_split(dataset=dataset)