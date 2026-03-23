import os

import torch.nn as nn
from torch.autograd import Variable
from types import SimpleNamespace

from tqdm import tqdm
from .aug_base import *
import torch.multiprocessing as mp
from math import ceil
import torch
import numpy as np
import torch.nn.functional as F
import gco

# Implementation Credit: https://github.com/snu-mllab/Co-Mixup/blob/main/main.py


@dataclass(kw_only=True)
class CoMixup(AugmentBase):

    def __post_init__(self):
        self.criterion_batch = torch.nn.CrossEntropyLoss(reduction="none")
        args = SimpleNamespace()

        default_args = [
            ("m_part", 20),
            ("m_block_num", 4),
            ("m_beta", 0.32),
            ("m_gamma", 1.0),
            ("m_thres", 0.83),
            ("m_thres_type", "hard"),
            ("m_eta", 0.05),
            ("mixup_alpha", 2.0),
            ("m_omega", 0.001),
            ("set_resolve", True),
            ("m_niter", 1),
            ("parallel", False),
            ("clean_lam", 1.0),
            ('num_classes', self.num_classes)
        ]

        for da in default_args:
            arg_key, default_v = da
            setattr(args, arg_key, self.config.get(arg_key, default_v))


        self.args = args
        print("CoMixup Args:", self.args)
        # self.mpp = None
        if args.parallel:
            batch_size = self.config.get('batch_size')
            if batch_size is None:
                raise RuntimeError("Please provide batch_size explicitly in CoMixup's config when using the parallel mode.")
            args.batch_size = batch_size
            print("CoMixup use parallel mode.")
            self.mpp = MixupProcessParallel(args.m_part, args.batch_size, 1)
        else:
            self.mpp = None

    def setup_based_on_model(self, setup_args: dict):
        training_model = setup_args.get('training_model')
        self.optimizer = setup_args.get('training_optimizer')
        # Use uncompiled backbone for saliency to avoid CUDAGraph conflicts
        inner = getattr(training_model, 'model', training_model)
        eager = getattr(inner, '_orig_mod', inner)
        # Wrap in a simple module that mimics LightningModule interface (forward, .device)
        self.saliency_model = _EagerModelWrapper(eager, training_model)

    def __call__(self, _x, _y) -> AugResult:
        # if y is one hot
        if torch.rand(1) <= self.config.get("prob", 1.0):
            x, y = self.co_mixup(
                _x,
                _y,
                self.saliency_model,
                self.optimizer,
                self.criterion_batch,
                self.args,
                mpp=self.mpp
            )
        return AugResult(
            augmented_x=x,
            augmented_y=y
        )

    def co_mixup(self, input, target, model, optimizer, criterion_batch, args, mpp=None):
        device = input.device
        criterion_batch = criterion_batch.to(device)
        sc = None

        input_var = Variable(input, requires_grad=True)
        target_var = Variable(target)
        A_dist = None

        # Calculate saliency (unary)
        if args.clean_lam == 0:
            model.eval()
            output = model(input_var)
            loss_batch = criterion_batch(output, target_var)
        else:
            model.train()
            output = model(input_var)
            loss_batch = 2 * args.clean_lam * criterion_batch(output,
                                                              target_var) / args.num_classes
        loss_batch_mean = torch.mean(loss_batch, dim=0)
        loss_batch_mean.backward()
        sc = torch.sqrt(torch.mean(input_var.grad**2, dim=1))

        batch_size = input.shape[0]

        # Here, we calculate distance between most salient location (Compatibility)
        # We can try various measurements
        with torch.no_grad():
            z = F.avg_pool2d(sc, kernel_size=8, stride=1)
            z_reshape = z.reshape(batch_size, -1)
            z_idx_1d = torch.argmax(z_reshape, dim=1)
            z_idx_2d = torch.zeros((batch_size, 2), device=device)
            z_idx_2d[:, 0] = z_idx_1d // z.shape[-1]
            z_idx_2d[:, 1] = z_idx_1d % z.shape[-1]
            A_dist = distance(z_idx_2d, dist_type='l1')

        if args.clean_lam == 0:
            model.train()
            optimizer.zero_grad()

        # Perform mixup and calculate loss
        target_reweighted = to_one_hot(target, args.num_classes, device)
        if args.parallel:
            out, target_reweighted = mpp(input.cpu(),
                                         target_reweighted.cpu(),
                                         args=args,
                                         sc=sc.cpu(),
                                         A_dist=A_dist.cpu())
            out = out.to(device)
            target_reweighted = target_reweighted.to(device)

        else:
            out, target_reweighted = mixup_process(input,
                                                   target_reweighted,
                                                   device,
                                                   args=args,
                                                   sc=sc,
                                                   A_dist=A_dist)

        return out, target_reweighted


class _EagerModelWrapper:
    """Thin wrapper around an uncompiled model that exposes .device, .eval(), .train(), and __call__."""

    def __init__(self, eager_model, lightning_module):
        self._model = eager_model
        self._lm = lightning_module

    @property
    def device(self):
        return self._lm.device

    def __call__(self, x):
        return self._model(x)

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


def to_one_hot(inp, num_classes, device=None):
    y_onehot = torch.zeros((inp.size(0), num_classes),
                           dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)

    return y_onehot


def distance(z, dist_type='l2'):
    '''Return distance matrix between vectors'''
    with torch.no_grad():
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        if dist_type[:2] == 'l2':
            A_dist = (diff**2).sum(-1)
            if dist_type == 'l2':
                A_dist = torch.sqrt(A_dist)
            elif dist_type == 'l22':
                pass
        elif dist_type == 'l1':
            A_dist = diff.abs().sum(-1)
        elif dist_type == 'linf':
            A_dist = diff.abs().max(-1)[0]
        else:
            return None
    return A_dist


def random_initialize(n_input, n_output, height, width):
    '''Initialization of labeling for Co-Mixup'''
    return np.random.randint(0, n_input, (n_output, width, height))


def to_onehot(idx, n_input, device=None):
    '''Return one-hot vector'''
    idx_onehot = torch.zeros(
        (idx.shape[0], n_input), dtype=torch.float32, device=device)
    idx_onehot.scatter_(1, idx.unsqueeze(1), 1)
    return idx_onehot


def obj_fn(cost_matrix, mask_onehot, beta, gamma):
    '''Calculate objective without thresholding'''
    n_output, height, width, n_input = mask_onehot.shape
    mask_idx_sum = mask_onehot.reshape(
        n_output, height * width, n_input).sum(1)

    loss = 0
    loss += torch.sum(cost_matrix.permute(1, 2, 0).unsqueeze(0) * mask_onehot)
    loss += beta / 2 * (((mask_onehot[:, :-1, :, :] - mask_onehot[:, 1:, :, :])**2).sum() +
                        ((mask_onehot[:, :, :-1, :] - mask_onehot[:, :, 1:, :])**2).sum())
    loss += gamma * (torch.sum(mask_idx_sum.sum(0)**2) -
                     torch.sum(mask_idx_sum**2))

    return loss


def graphcut_multi(cost, beta=1, algorithm='swap', n_label=0, add_idx=None):
    '''find optimal labeling using Graph-Cut algorithm'''
    height, width, n_input = cost.shape

    unary = np.ascontiguousarray(cost)
    pairwise = (np.ones(shape=(n_input, n_input), dtype=np.float32) -
                np.eye(n_input, dtype=np.float32))
    if n_label == 2:
        pairwise[-1, :-1][add_idx] = 0.25
        pairwise[:-1, -1][add_idx] = 0.25
    elif n_label == 3:
        pairwise[-3:, :-3][:, add_idx] = np.array([[0.25, 0.25, 1], [0.25, 1, 0.25],
                                                   [1, 0.25, 0.25]])
        pairwise[:-3, -3:][add_idx, :] = np.array([[0.25, 0.25, 1], [0.25, 1, 0.25],
                                                   [1, 0.25, 0.25]])

    cost_v = beta * np.ones(shape=[height - 1, width], dtype=np.float32)
    cost_h = beta * np.ones(shape=[height, width - 1], dtype=np.float32)

    mask_idx = gco.cut_grid_graph(
        unary, pairwise, cost_v, cost_h, algorithm=algorithm)
    return mask_idx


def graphcut_wrapper(cost_penalty, label_count, n_input, height, width, beta, device, iter_idx=0):
    '''Wrapper of graphcut_multi performing efficient extension to multi-label'''
    assigned_label = (label_count > 0)
    if iter_idx > 0:
        n_label = int(assigned_label.float().sum())
    else:
        n_label = 0

    if n_label == 2:
        cost_add = cost_penalty[:, :,
                                assigned_label].mean(-1, keepdim=True) - 5e-4
        cost_penalty = torch.cat([cost_penalty, cost_add], dim=-1)
        unary = cost_penalty.cpu().numpy()

        mask_idx_np = graphcut_multi(unary,
                                     beta=beta,
                                     n_label=2,
                                     add_idx=assigned_label.cpu().numpy(),
                                     algorithm='swap')
        mask_idx_onehot = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                    n_input + 1,
                                    device=device).reshape(height, width, n_input + 1)

        idx_matrix = torch.zeros([1, 1, n_input], device=device)
        idx_matrix[:, :, assigned_label] = 0.5
        mask_onehot_i = mask_idx_onehot[:, :, :n_input] + mask_idx_onehot[:, :,
                                                                          n_input:] * idx_matrix
    elif n_label >= 3:
        soft_label = torch.tensor(
            [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], device=device)

        _, indices = torch.topk(label_count, k=3)
        assigned_label = torch.zeros_like(assigned_label)
        assigned_label[indices] = True

        cost_add = torch.matmul(
            cost_penalty[:, :, assigned_label], soft_label) - 5e-4
        cost_penalty = torch.cat([cost_penalty, cost_add], dim=-1)
        unary = cost_penalty.cpu().numpy()

        mask_idx_np = graphcut_multi(unary,
                                     beta=beta,
                                     n_label=3,
                                     add_idx=assigned_label.cpu().numpy(),
                                     algorithm='swap')
        mask_idx_onehot = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                    n_input + 3,
                                    device=device).reshape(height, width, n_input + 3)

        idx_matrix = torch.zeros([3, n_input], device=device)
        idx_matrix[:, assigned_label] = soft_label
        mask_onehot_i = mask_idx_onehot[:, :, :n_input] + torch.matmul(
            mask_idx_onehot[:, :, n_input:], idx_matrix)
    else:
        unary = cost_penalty.cpu().numpy()
        mask_idx_np = graphcut_multi(unary, beta=beta, algorithm='swap')
        mask_onehot_i = to_onehot(torch.tensor(mask_idx_np, device=device, dtype=torch.long),
                                  n_input,
                                  device=device).reshape(height, width, n_input)

    return mask_onehot_i


def resolve_label(assigned_label_total, device=None):
    '''A post-processing for resolving identical outputs'''
    n_output, n_input = assigned_label_total.shape
    add_cost = torch.zeros_like(assigned_label_total)

    dist = torch.min(
        (assigned_label_total.unsqueeze(1) -
         assigned_label_total.unsqueeze(0)).abs().sum(-1),
        torch.tensor(1.0, device=device))
    coincide = torch.triu(1. - dist, diagonal=1)

    for i1, i2 in coincide.nonzero():
        nonzeros = assigned_label_total[i1].nonzero()
        if len(nonzeros) == 1:
            continue
        else:
            add_cost[i1][nonzeros[0]] = 1.
            add_cost[i2][nonzeros[1]] = 1.

    return add_cost


def get_onehot_matrix(cost_matrix,
                      A,
                      n_output,
                      idx=None,
                      beta=0.32,
                      gamma=1.,
                      eta=0.05,
                      mixup_alpha=2.0,
                      thres=0.84,
                      thres_type='hard',
                      set_resolve=True,
                      niter=3,
                      device=None):
    '''Iterative submodular minimization algorithm with the modularization of supermodular term'''
    n_input, height, width = cost_matrix.shape
    thres = thres * height * width
    beta = beta / height / width
    gamma = gamma / height / width
    eta = eta / height / width

    add_cost = None

    # Add prior term
    lam = mixup_alpha * torch.ones(n_input, device=device)
    alpha = torch.distributions.dirichlet.Dirichlet(
        lam).sample().reshape(n_input, 1, 1)
    cost_matrix -= eta * torch.log(alpha + 1e-8)

    with torch.no_grad():
        # Init
        if idx is None:
            mask_idx = torch.tensor(random_initialize(n_input, n_output, height, width),
                                    device=device)
        else:
            mask_idx = idx

        mask_onehot = to_onehot(mask_idx.reshape(-1), n_input,
                                device=device).reshape([n_output, height, width, n_input])

        loss_prev = obj_fn(cost_matrix, mask_onehot, beta, gamma)
        penalty = to_onehot(mask_idx.reshape(-1), n_input,
                            device=device).sum(0).reshape(-1, 1, 1)

        # Main loop
        for iter_idx in range(niter):
            for i in range(n_output):
                label_count = mask_onehot[i].reshape(
                    [height * width, n_input]).sum(0)
                penalty -= label_count.reshape(-1, 1, 1)
                if thres_type == 'hard':
                    modular_penalty = (2 * gamma * (
                        (A @ penalty.squeeze() > thres).float() * A @ penalty.squeeze())).reshape(
                            -1, 1, 1)
                elif thres_type == 'soft':
                    modular_penalty = (2 * gamma * ((A @ penalty.squeeze() > thres).float() *
                                                    (A @ penalty.squeeze() - thres))).reshape(
                                                        -1, 1, 1)
                else:
                    raise AssertionError("wrong threshold type!")

                if add_cost is not None:
                    cost_penalty = (cost_matrix + modular_penalty +
                                    gamma * add_cost[i].reshape(-1, 1, 1)).permute(1, 2, 0)
                else:
                    cost_penalty = (
                        cost_matrix + modular_penalty).permute(1, 2, 0)

                mask_onehot[i] = graphcut_wrapper(cost_penalty, label_count, n_input, height, width,
                                                  beta, device, iter_idx)
                penalty += mask_onehot[i].reshape([height * width,
                                                   n_input]).sum(0).reshape(-1, 1, 1)

            if iter_idx == niter - 2 and set_resolve:
                assigned_label_total = (mask_onehot.reshape(n_output, -1, n_input).sum(1) >
                                        0).float()
                add_cost = resolve_label(assigned_label_total, device=device)

            loss = obj_fn(cost_matrix, mask_onehot, beta, gamma)
            if (loss_prev - loss).abs() / loss.abs() < 1e-6:
                break
            loss_prev = loss

    return mask_onehot


def mix_input(mask_onehot, input_sp, target_reweighted, sc=None):
    ''' Mix inputs and one-hot labels based on labeling (mask_onehot)'''
    n_output, height, width, n_input = mask_onehot.shape
    _, n_class = target_reweighted.shape

    mask_onehot_im = F.interpolate(mask_onehot.permute(0, 3, 1, 2),
                                   size=input_sp.shape[-1],
                                   mode='nearest')
    output = torch.sum(mask_onehot_im.unsqueeze(2) *
                       input_sp.unsqueeze(0), dim=1)

    if sc is None:
        mask_target = torch.matmul(mask_onehot, target_reweighted)
    else:
        weighted_mask = mask_onehot * sc.permute(1, 2, 0).unsqueeze(0)
        mask_target = torch.matmul(weighted_mask, target_reweighted)

    target = mask_target.reshape(n_output, height * width, n_class).sum(-2)
    target /= target.sum(-1, keepdim=True)

    return output, target


def mixup_process(out, target_reweighted, device, args=None, sc=None, A_dist=None):
    m_block_num = args.m_block_num
    m_part = args.m_part
    batch_size = out.shape[0]
    width = out.shape[-1]

    if A_dist is None:
        A_dist = torch.eye(batch_size, device=out.device)

    if m_block_num == -1:
        m_block_num = 2**np.random.randint(1, 5)

    block_size = width // m_block_num
    sc = F.avg_pool2d(sc, block_size)

    out_list = []
    target_list = []

    # Partition a batch
    for i in range(ceil(batch_size / m_part)):
        with torch.no_grad():
            sc_part = sc[i * m_part:(i + 1) * m_part]
            A_dist_part = A_dist[i * m_part:(i + 1)
                                 * m_part, i * m_part:(i + 1) * m_part]

            n_input = sc_part.shape[0]
            sc_norm = sc_part / \
                sc_part.reshape(n_input, -1).sum(1).reshape(n_input, 1, 1)
            cost_matrix = -sc_norm

            A_base = torch.eye(n_input, device=out.device)
            A_dist_part = A_dist_part / torch.sum(A_dist_part) * n_input
            A = (1 - args.m_omega) * A_base + args.m_omega * A_dist_part

            # Return a batch(partitioned) of mixup labeling
            mask_onehot = get_onehot_matrix(cost_matrix.detach(),
                                            A,
                                            n_output=n_input,
                                            beta=args.m_beta,
                                            gamma=args.m_gamma,
                                            eta=args.m_eta,
                                            mixup_alpha=args.mixup_alpha,
                                            thres=args.m_thres,
                                            thres_type=args.m_thres_type,
                                            set_resolve=args.set_resolve,
                                            niter=args.m_niter,
                                            # device='cuda')
                                            device=device)

        # Generate image and corrsponding soft target
        output_part, target_part = mix_input(mask_onehot, out[i * m_part:(i + 1) * m_part],
                                             target_reweighted[i * m_part:(i + 1) * m_part])

        out_list.append(output_part)
        target_list.append(target_part)

    with torch.no_grad():
        out = torch.cat(out_list, dim=0)
        target_reweighted = torch.cat(target_list, dim=0)

    return out.contiguous(), target_reweighted


def mixup_process_worker(out: torch.Tensor,
                         target_reweighted: torch.Tensor,
                         hidden=0,
                         args=None,
                         sc: torch.Tensor = None,
                         A_dist: torch.Tensor = None,
                         debug=False):
    """Perform Co-Mixup"""
    m_block_num = args.m_block_num
    n_input = out.shape[0]
    width = out.shape[-1]

    if m_block_num == -1:
        m_block_num = 2**np.random.randint(1, 5)

    block_size = width // m_block_num

    with torch.no_grad():
        if A_dist is None:
            A_dist = torch.eye(n_input, device=out.device)
        A_base = torch.eye(n_input, device=out.device)

        sc = F.avg_pool2d(sc, block_size)
        sc_norm = sc / sc.view(n_input, -1).sum(1).view(n_input, 1, 1)
        cost_matrix = -sc_norm

        A_dist = A_dist / torch.sum(A_dist) * n_input
        A = (1 - args.m_omega) * A_base + args.m_omega * A_dist

        # Return a batch(partitioned) of mixup labeling
        mask_onehot = get_onehot_matrix(cost_matrix.detach(),
                                        A,
                                        n_output=n_input,
                                        beta=args.m_beta,
                                        gamma=args.m_gamma,
                                        eta=args.m_eta,
                                        mixup_alpha=args.mixup_alpha,
                                        thres=args.m_thres,
                                        thres_type=args.m_thres_type,
                                        set_resolve=args.set_resolve,
                                        niter=args.m_niter,
                                        device=out.device)
        # Generate image and corrsponding soft target
        out, target_reweighted = mix_input(mask_onehot, out, target_reweighted)

    return out.contiguous(), target_reweighted


def mixup_process_worker_wrapper(q_input: mp.Queue, q_output: mp.Queue, gpu_id: int):
    """
    :param q_input:		input queue
    :param q_output:	output queue
    :param gpu_id:		running gpu device id
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    print(f"Process generated with cuda:{gpu_id}")
    device = torch.device(f"cuda:{gpu_id}")
    while True:
        # Get and load on gpu
        out, target_reweighted, hidden, args, sc, A_dist, debug = q_input.get()
        out = out.to(device)
        target_reweighted = target_reweighted.to(device)
        sc = sc.to(device)
        A_dist = A_dist.to(device)

        # Run
        out, target_reweighted = mixup_process_worker(out, target_reweighted, hidden, args, sc,
                                                      A_dist, debug)
        # To cpu and return
        out = out.cpu()
        target_reweighted = target_reweighted.cpu()
        q_output.put([out, target_reweighted])


class MixupProcessWorker:
    def __init__(self, device: int):
        """
        :param device: gpu device id
        """
        ctx = mp.get_context('spawn')
        self.q_input = ctx.Queue()
        self.q_output = ctx.Queue()
        self.worker = ctx.Process(target=mixup_process_worker_wrapper,
                                  args=[self.q_input, self.q_output, device])
        self.worker.deamon = True
        self.worker.start()

    def start(self,
              out: torch.Tensor,
              target_reweighted: torch.Tensor,
              hidden=0,
              args=None,
              sc: torch.Tensor = None,
              A_dist: torch.Tensor = None,
              debug=True):
        self.q_input.put(
            [out, target_reweighted, hidden, args, sc, A_dist, debug])

    def join(self):
        input, target = self.q_output.get()
        return input, target

    def close(self):
        self.worker.terminate()


class MixupProcessParallel:
    def __init__(self, part, batch_size, num_gpu=1):
        """
        :param part:
        :param batch_size:
        :param num_gpu:
        """
        self.part = part
        self.batch_size = batch_size
        self.n_workers = ceil(batch_size / part)
        self.workers = [MixupProcessWorker(
            device=i % num_gpu) for i in range(self.n_workers)]

    def __call__(self,
                 out: torch.Tensor,
                 target_reweighted: torch.Tensor,
                 hidden=0,
                 args=None,
                 sc: torch.Tensor = None,
                 A_dist: torch.Tensor = None,
                 debug=False):
        '''
        :param out:					cpu tensor
        :param target_reweighted: 	cpu tensor
        :param hidden:
        :param args:				cpu args
        :param sc: 					cpu tensor
        :param A_dist: 				cpu tensor
        :param debug:
        :return:					out, target_reweighted (cpu tensor)
        '''

        for idx in range(self.n_workers):
            self.workers[idx].start(
                out[idx * self.part:(idx + 1) * self.part].contiguous(),
                target_reweighted[idx * self.part:(idx + 1)
                                  * self.part].contiguous(), hidden, args,
                sc[idx * self.part:(idx + 1) * self.part].contiguous(),
                A_dist[idx * self.part:(idx + 1) * self.part,
                       idx * self.part:(idx + 1) * self.part].contiguous(), debug)
        # join
        out_list = []
        target_list = []
        for idx in range(self.n_workers):
            out, target = self.workers[idx].join()
            out_list.append(out)
            target_list.append(target)

        return torch.cat(out_list), torch.cat(target_list)

    def close(self):
        for w in self.workers:
            w.close()


if __name__ == "__main__":
    '''unit test'''
    mp.set_start_method("spawn")

    # inputs (cpu) : out0, target_reweighted0, out, target_reweighted, args, sc, A_dist
    d = torch.load("input.pt")
    out0 = d["out0"]
    target_reweighted0 = d["target_reweighted0"]
    args = d["args"]
    sc = d["sc"]
    A_dist = d["A_dist"]

    # Parallel mixup wrapper
    mpp = MixupProcessParallel(args.m_part, args.batch_size, num_gpu=1)

    # For cuda initialize
    torch.ones(3).cuda()
    for iter in tqdm(range(1), desc="initialize"):
        out, target_reweighted = mpp(out0,
                                     target_reweighted0,
                                     args=args,
                                     sc=sc,
                                     A_dist=A_dist,
                                     debug=True)

    # Parallel run
    for iter in tqdm(range(100), desc="parallel"):
        out, target_reweighted = mpp(out0,
                                     target_reweighted0,
                                     args=args,
                                     sc=sc,
                                     A_dist=A_dist,
                                     debug=True)

    print((d["out"].cpu() == out.cpu()).float().mean())
    print((d["target_reweighted"].cpu() ==
          target_reweighted.cpu()).float().mean())

    # Original run
    out0cuda = out0.cuda()
    target_reweighted0cuda = target_reweighted0.cuda()
    sccuda = sc.cuda()
    A_distcuda = A_dist.cuda()
    for iter in tqdm(range(100), desc="original"):
        out, target_reweighted = mixup_process(out0cuda,
                                               target_reweighted0cuda,
                                               args=args,
                                               sc=sccuda,
                                               A_dist=A_distcuda,
                                               debug=True)

    print((d["out"].cpu() == out.cpu()).float().mean())
    print((d["target_reweighted"].cpu() ==
          target_reweighted.cpu()).float().mean())

    print("end")
