from dataclasses import dataclass
from typing import NamedTuple

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
import torch.nn.functional as F
Device = torch.device
@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                    self.pad_vocab_size_multiple
                    - self.vocab_size % self.pad_vocab_size_multiple
            )


class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )


class Mamba2(nn.Module):
    def __init__(self, d_model: int,  # model dimension (D)
                 n_layer: int = 24,  # number of Mamba-2 layers in the language model
                 d_state: int = 128,  # state dimension (N)
                 d_conv: int = 4,  # convolution kernel size
                 expand: int = 2,  # expansion factor (E)
                 headdim: int = 64,  # head dimension (P)
                 chunk_size: int = 64,  # matrix partition size (Q)
                 vocab_size: int = 50277,
                 pad_vocab_size_multiple: int = 16, ):
        super().__init__()
        args = Mamba2Config(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, vocab_size, pad_vocab_size_multiple)
        self.args = args
        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, ))
        self.A_log = nn.Parameter(torch.empty(args.nheads, ))
        self.D = nn.Parameter(torch.empty(args.nheads, ))
        self.norm = RMSNorm(args.d_inner, )
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, )

    def forward(self, u: Tensor, h=None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        if h:
            return self.step(u, h)

        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )

        xBC = silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=x.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(self, u: Tensor, h: InferenceCache):
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * torch.sigmoid(x)
class BaseNdMamba2(nn.Module):
    def __init__(self, cin, cout, mamba_dim, **mamba2_args):
        super().__init__()
        assert mamba_dim % 64 == 0, "cmid 必须是64的倍数"
        self.fc_in = nn.Linear(cin, mamba_dim, bias=False)  # 调整通道数到cmid
        self.mamba2_for = Mamba2(mamba_dim, **mamba2_args)  # 正向
        self.mamba2_back = Mamba2(mamba_dim, **mamba2_args)  # 负向
        self.fc_out = nn.Linear(mamba_dim, cout, bias=False)  # 调整通道数到cout


# NdMamba2 类
class NdMamba2(BaseNdMamba2):
    def __init__(self, cin, cout, mamba_dim, **mamba2_args):
        super().__init__(cin, cout, mamba_dim, **mamba2_args)

    def forward(self, x):
        #print(f"Input shape to NdMamba2: {x.shape}")

        if x.dim() == 2:
            x = torch.flatten(x, start_dim=1)  # 展平从第 1 维开始
        else:
            x = torch.flatten(x, start_dim=2)  # 从第 2 维开始

        #print(f"Shape after initial flatten: {x.shape}")

        size = x.shape[2:]
        x = torch.flatten(x, 2)  # b c size
        l = x.shape[2]
        x = F.pad(x, (0, (64 - x.shape[2] % 64) % 64))  # 将 l , pad到4的倍数, [b, c64,l4]
        #print(f"Shape after padding: {x.shape}")

        x = rearrange(x, 'b c l -> b l c')  # 转成 1d 信号 [b, d4*w4*h4, c64]
        x = self.fc_in(x)  # 调整通道数为目标通道数

        x1, h1 = self.mamba2_for(x)
        x2, h2 = self.mamba2_back(x.flip(1))#flip将x的维度进行翻转
        x2 = x2.flip(1)

        x = x1 + x2
        x = self.fc_out(x)  # 调整通道数为目标通道数

        x = rearrange(x, 'b l c -> b c l')  # 转成 2d 图片[b, l64, c64]
        x = x[:, :, :l]  # 截取原图大小
        x = x.view(x.size(0), x.size(1), *size)

        #print(f"Output shape from NdMamba2: {x.shape}")  # 打印输出形状
        return x

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self, oup_channels: int, group_num: int = 16, gate_treshold: float = 0.5):
        super().__init__()
        self.gn = GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        reweights = self.sigmoid(gn_x * w_gamma.view(1, -1, 1, 1))

        info_mask = reweights >= self.gate_treshold
        noninfo_mask = reweights < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0 < alpha < 1
    '''

    def __init__(self, op_channel: int, alpha: float = 1 / 2, squeeze_radio: int = 2, group_size: int = 2,
                 group_kernel_size: int = 3):
        super().__init__()
        self.up_channel = int(alpha * op_channel)
        self.low_channel = op_channel - self.up_channel
        self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, self.low_channel // squeeze_radio, kernel_size=1, bias=False)

        # Up
        self.GWC = nn.Conv2d(self.up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size,
                             padding=group_kernel_size // 2)
        self.PWC1 = nn.Conv2d(self.up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)

        # Low
        self.PWC2 = nn.Conv2d(self.low_channel // squeeze_radio, op_channel - (self.low_channel // squeeze_radio),
                              kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)

        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv_mamba(nn.Module):
    def __init__(self, label_num):
        super(ScConv_mamba, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 25), 1, padding='same'),
            nn.BatchNorm2d(32),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d((1, 3), 3, padding=(0, 1)),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 25), 1, padding='same'),
            nn.BatchNorm2d(64),  # 添加 Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d((1, 3), 3, padding=(0, 1))
        )

        # 新增的 NdMamba2 模块
        self.nd_mamba2 = NdMamba2(cin=64, cout=64, mamba_dim=128)  # 根据需求设置 cin, cout 和 mamba_dim

        # SRU 和 CRU 层
        self.sru = SRU(oup_channels=64)
        self.cru = CRU(op_channel=64)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5632, 1024),  # 使用动态形状
            nn.Dropout(p=0.5),
            nn.Linear(1024, label_num),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        x = self.layer_1(x)
        #print(f'After layer 1: {x.shape}')  # 打印形状
        x = self.layer_2(x)
        #print(f'After layer 2: {x.shape}')  # 打印形状

        x = self.sru(x)
        #print(f'After SRU: {x.shape}')  # 打印形状
        x = self.cru(x)
        #print(f'After CRU: {x.shape}')  # 打印形状

        # 将输出传递给 NdMamba2
        x = self.nd_mamba2(x)
        #print(f'After NdMamba2: {x.shape}')  # 打印形状

        # 使用 reshape 代替 view
        if x.dim() == 3:
            n_features = x.size(1) * x.size(2)
        elif x.dim() == 4:
            n_features = x.size(1) * x.size(2) * x.size(3)
        else:
            raise ValueError(f"Unexpected tensor shape: {x.shape}")

        # 使用 reshape 来调整
        x = x.reshape(x.size(0), -1)  # 替代 view 处理

        x = self.fc1(x)  # 经过全连接层
        return x


if __name__ == "__main__":
    model = ScConv_mamba(label_num=12)
    x = torch.randn(128, 1, 1, 50)  # 假设输入大小为 (1, 1, 1, 200)
    output = model(x)
    print(output.shape)