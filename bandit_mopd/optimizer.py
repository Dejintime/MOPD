"""Full-parameter AdamW with FP32 master weights/moments on CPU.

GPU gradients and BF16 parameters remain on the student GPU. A parameter is
transferred in chunks so no full CPU gradient mirror is allocated. Numerical
behavior is tested against torch.optim.AdamW on FP32 witnesses.
"""
import math
import torch


class CPUAdamW:
    def __init__(self, parameters, lr=2e-6, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, chunk_size=1_000_000):
        self.params = [p for p in parameters if p.requires_grad]
        self.lr, self.betas, self.eps = lr, betas, eps
        self.weight_decay, self.chunk_size = weight_decay, chunk_size
        self.state = {}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def initialize_state(self):
        """Allocate persistent FP32 CPU state without a full FP32 GPU copy."""
        for index, p in enumerate(self.params):
            if index not in self.state:
                master = p.detach().to(device='cpu', dtype=torch.float32, copy=True).contiguous().view(-1)
                self.state[index] = {'master':master, 'm':torch.zeros_like(master),
                                     'v':torch.zeros_like(master), 'step':0}

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for index, p in enumerate(self.params):
            if p.grad is None:
                continue
            if index not in self.state:
                master = p.detach().to(device='cpu', dtype=torch.float32, copy=True).contiguous().view(-1)
                self.state[index] = {"master": master, "m": torch.zeros_like(master),
                                     "v": torch.zeros_like(master), "step": 0}
            state = self.state[index]
            state["step"] += 1
            t = state["step"]
            pflat, gflat = p.view(-1), p.grad.view(-1)
            for start in range(0, p.numel(), self.chunk_size):
                sl = slice(start, start + self.chunk_size)
                g = gflat[sl].float().cpu()
                master, m, v = (state[key][sl] for key in ("master", "m", "v"))
                m.mul_(b1).add_(g, alpha=1-b1)
                v.mul_(b2).addcmul_(g, g, value=1-b2)
                master.mul_(1 - self.lr * self.weight_decay)
                denom = v.sqrt().div_(math.sqrt(1-b2**t)).add_(self.eps)
                master.addcdiv_(m, denom, value=-self.lr/(1-b1**t))
                pflat[sl].copy_(master)

    def state_dict(self):
        return {"state": self.state, "lr": self.lr, "betas": self.betas,
                "eps": self.eps, "weight_decay": self.weight_decay}
