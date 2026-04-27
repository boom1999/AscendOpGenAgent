## 功能说明

- 接口功能：输入qkv融合张量，通过SplitVD拆分q、k、v张量，执行RmsNorm、ApplyRotaryPosEmb、Quant、Scatter融合操作，输出q_out、k_cache、v_cache、q_out_before_quant(可选)、k_out_before_quant(可选)、v_out_before_quant(可选)。
- 本接口目前支持的场景如下表：

  |场景类型|情况概要|
  |:---|:---|
  |<ul><li>cache_mode为PA_NZ</li><li>q无量化</li><li>k和v支持无量化、对称量化和非对称量化</li><li>q_out_before_quant/k_out_before_quant/v_out_before_quant不输出</li></ul>|qkv Shape为[$B_{qkv}$ * $S_{qkv}$, $N_{qkv}$ * $D_{qkv}$]，q、k、v具有完全相同的D维度。主要计算过程与输出对应关系：<br><ul><li>qkv 经过SplitVD->q、k、v</li><li>q经过RmsNorm、RoPE->q_out</li><li>k经过RmsNorm、RoPE、Quant(可选)、Scatter->k_cache</li><li>v经过Quant(可选)、Scatter->v_cache</li></ul>|

## 计算公式

  (1) SplitVD:

  下式中，$N_q$、$N_k$、$N_v$分别表示 q、k、v 分量的注意力头数量，必须满足：

  $$
  \begin{cases}
  N_k = N_v \\
  N_{qkv} = N_k + N_v + N_q \\
  D_{qkv} = D_q = D_k = D_v
  \end{cases}
  $$

  $$
  \begin{aligned}
  q &= qkv[..., [:N_q] * D_{qkv}] \\
  k &= qkv[..., [N_q:-N_v] * D_{qkv}] \\
  v &= qkv[..., [-N_v:] * D_{qkv}]
  \end{aligned}
  $$

  (2) RmsNorm:

  此处x和y分别表示RmsNorm的输入张量和输出张量，归一化沿最后一维（feature dimension）进行，该计算规则通用于q、k分量。

  $$
  squareX = x * x
  $$

  $$
  meanSquareX = squareX.mean(dim = -1, keepdim = True)
  $$

  $$
  rms = \sqrt{meanSquareX + epsilon}
  $$

  $$
  y = (x / rms) * gamma
  $$

  (3) RoPE (Half-and-Half):

  此处的y指代完成RmsNorm计算的输出结果。

  $$
  y1 = y[\ldots, :d/2]
  $$

  $$
  y2 = y[\ldots, d/2:]
  $$

  $$
  y\_RoPE = torch.cat((-y2, y1), dim = -1)
  $$
  
  $$
  y\_embed = (y * cos) + y\_RoPE * sin
  $$

  (4) Quant:

  无量化：

  $$
  kQuant = kRoPE \\
  vQuant = v
  $$

  对称量化部分：

  $$
  kQuant = kRoPE / kScale \\
  vQuant = v / vScale
  $$

  非对称量化部分：
  
  $$
  kQuant = kRoPE / kScale + kOffset \\
  vQuant = v / vScale + vOffset
  $$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| qkv | 输入 | 用于切分出q、k、v的输入数据 | FLOAT16、BFLOAT16 | [B_qkv * S_qkv, N_qkv * D_qkv] |
| q_gamma | 输入 | q的rms_norm缩放参数 | FLOAT16、BFLOAT16 | [D_qkv] |
| k_gamma | 输入 | k的rms_norm缩放参数 | FLOAT16、BFLOAT16 | [D_qkv] |
| cos | 输入 | rope计算的余弦变换输入 | FLOAT16、BFLOAT16 | [B_qkv * S_qkv, 1 * D_rope]，D_rope=D_qkv |
| sin | 输入 | rope计算的正弦变换输入 | FLOAT16、BFLOAT16 | 同cos |
| index | 输入 | 指定写入cache的具体索引位置 | INT64 | [B_qkv * S_qkv] |
| q_out | 输入/输出 | 提前申请的cache，输入输出同地址复用 | FLOAT16、BFLOAT16 | [B_qkv * S_qkv, N_q * D_qkv] |
| k_cache | 输入/输出 | 提前申请的cache，不量化时同qkv dtype，量化时INT8 | FLOAT16、BFLOAT16、INT8 | 不量化：[BlockNum, N_k*D_qkv//16, BlockSize, 16]；量化：[BlockNum, N_k*D_qkv//32, BlockSize, 32] |
| v_cache | 输入/输出 | 提前申请的cache，不量化时同qkv dtype，量化时INT8 | FLOAT16、BFLOAT16、INT8 | 不量化：[BlockNum, N_v*D_qkv//16, BlockSize, 16]；量化：[BlockNum, N_v*D_qkv//32, BlockSize, 32] |
| k_scale | 可选输入 | k的量化缩放因子（k_cache为INT8时需要） | FLOAT32 | [N_k, D_qkv] |
| v_scale | 可选输入 | v的量化缩放因子（v_cache为INT8时需要） | FLOAT32 | [N_v, D_qkv] |
| k_offset | 可选输入 | k的非对称量化偏移量 | FLOAT32 | [N_k, D_qkv] |
| v_offset | 可选输入 | v的非对称量化偏移量 | FLOAT32 | [N_v, D_qkv] |
| q_out_before_quant | 可选输出 | 即将写入q_out中的数据 | FLOAT16、BFLOAT16 | 同q_out |
| k_out_before_quant | 可选输出 | 未经量化和Scatter前的k中间计算结果 | FLOAT16、BFLOAT16 | ND |
| v_out_before_quant | 可选输出 | 未经量化和Scatter前的v中间计算结果 | FLOAT16、BFLOAT16 | ND |
| qkv_size | 属性 | 按[B_qkv, S_qkv, N_qkv, D_qkv]顺序传入 | INT64 | - |
| head_nums | 属性 | 按[N_q, N_k, N_v]顺序传入 | INT64 | - |
| epsilon | 可选属性 | RmsNorm计算防除0，默认1e-6 | FLOAT32 | - |
| cache_mode | 可选属性 | cache格式选择标记，目前只支持PA_NZ，默认PA_NZ | STRING | - |
| is_output_qkv | 可选属性 | 是否输出未经量化和Scatter前的原始值，默认false | BOOL | - |

## 约束说明

- 输入shape限制：
  - D_qkv目前仅支持128。D_q=D_k=D_v=D_qkv，且需满足（D_qkv*qkv数据类型占字节数）可以被32整除。
  - 根据rope规则，D_k和D_q为偶数。PA_NZ场景下，D_k、D_q需32B对齐；BlockSize需32B对齐。
  - 32B对齐的具体值由cache数据类型决定：若cache的dtype为int8，则BlockSize%32=0；若为float16，则BlockSize%16=0；若k_cache与v_cache的dtype不一致，BlockSize需同时满足两者。
  - BlockNum >= Ceil(S_qkv / BlockSize) * B_qkv。
- 其他限制：
  - index的value值范围为[-1, BlockNum * BlockSize)，值不可重复。index为-1时代表跳过更新。
  - k_scale、v_scale表示对称量化的缩放因子，若传参则值不能为0。

```python
class Model(nn.Module):
    """QKV RmsNorm + RoPE + Quant + Cache scatter."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        qkv: torch.Tensor,
        gamma_q: torch.Tensor,
        gamma_k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        index: torch.Tensor,
        q_out: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor = None,
        v_scale: torch.Tensor = None,
        k_offset: torch.Tensor = None,
        v_offset: torch.Tensor = None,
        qkv_size: list = None,
        head_nums: list = None,
        epsilon: float = 1e-6,
        cache_mode: str = "PA_NZ",
        is_output_qkv: bool = False,
    ) -> List[torch.Tensor]:
        """
        QKV fused RmsNorm + RoPE + Quant + Scatter.

        Splits qkv into q, k, v by head_nums.
        q: RmsNorm -> RoPE -> q_out
        k: RmsNorm -> RoPE -> Quant(optional) -> Scatter -> k_cache
        v: Quant(optional) -> Scatter -> v_cache

        Args:
            qkv: (B*S, Nqkv*D) fused input
            gamma_q, gamma_k: (D,) rmsNorm weights
            cos, sin: (B*S, Nrope*D) RoPE factors
            index: scatter indices
            q_out, k_cache, v_cache: output buffers
            k_scale, v_scale, k_offset, v_offset: optional quant params
            qkv_size: [B, S, Nqkv, D]
            head_nums: [Nq, Nk, Nv]
            epsilon: rmsNorm epsilon
            cache_mode: cache layout
            is_output_qkv: output pre-quant values
        Returns:
            List of [q_out, k_cache, v_cache, ...optional...]
        """
        if qkv_size is None:
            qkv_size = [1, qkv.shape[0], 1, qkv.shape[1]]
        if head_nums is None:
            head_nums = [1, 0, 0]

        qkv_dtype = qkv.dtype
        qkv_f = qkv.float()
        gamma_q_f = gamma_q.float()
        gamma_k_f = gamma_k.float()
        cos_f = cos.float()
        sin_f = sin.float()

        Bqkv, Sqkv, Nqkv, Dqkv = qkv_size
        Nq, Nk, Nv = head_nums

        q, k, v = qkv_f.split([Nq * Dqkv, Nk * Dqkv, Nv * Dqkv], dim=-1)
        q_4 = q.view(Bqkv, Sqkv, Nq, Dqkv)
        k_4 = k.view(Bqkv, Sqkv, Nk, Dqkv)
        v_4 = v.view(Bqkv, Sqkv, Nv, Dqkv)

        Srope = cos_f.shape[0] // Bqkv
        Nrope = cos_f.shape[1] // Dqkv
        cos_4 = cos_f.view(Bqkv, Srope, Nrope, Dqkv)
        sin_4 = sin_f.view(Bqkv, Srope, Nrope, Dqkv)

        def rms_norm(x, eps, gamma):
            return (x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)) * gamma

        def rope(rms_res, cos_t, sin_t, rope_range):
            B, S, N = rms_res.shape[:3]
            rope_dim = rope_range[1] - rope_range[0]
            Srope_l = cos_t.shape[1]
            Nrope_l = cos_t.shape[2]
            if Srope_l == 1:
                cos_t = cos_t.repeat(1, S, 1, 1)
                sin_t = sin_t.repeat(1, S, 1, 1)
            if Nrope_l == 1:
                cos_t = cos_t.repeat(1, 1, N, 1)
                sin_t = sin_t.repeat(1, 1, N, 1)
            rope_in = rms_res[..., rope_range[0]:rope_range[1]]
            r1 = rope_in[..., :rope_in.shape[-1] // 2]
            r2 = rope_in[..., rope_in.shape[-1] // 2:]
            rotate_half = torch.cat((-r2, r1), dim=-1)
            rope_embed = (rope_in * cos_t) + (rotate_half * sin_t)
            return torch.cat([rms_res[..., :rope_range[0]], rope_embed, rms_res[..., rope_range[1]:]], dim=-1)

        def quant(out, scale, offset):
            if scale is not None:
                out = out / scale
            if offset is not None:
                out = out + offset
            if scale is not None:
                out = torch.round(out).clamp(-128, 127)
            return out

        rope_range = [0, Dqkv]

        # q computation
        q_4 = rms_norm(q_4, epsilon, gamma_q_f)
        q_4 = rope(q_4, cos_4, sin_4, rope_range)
        q_out_res = q_4.reshape(Bqkv * Sqkv, Nq * Dqkv)

        # k computation
        k_4 = rms_norm(k_4, epsilon, gamma_k_f)
        k_4 = rope(k_4, cos_4, sin_4, rope_range)
        k_4 = quant(k_4, k_scale, k_offset)

        # v computation
        v_4 = quant(v_4, v_scale, v_offset)

        if is_output_qkv:
            return [q_out_res.to(qkv_dtype), k_cache, v_cache,
                    q_out_res.to(qkv_dtype),
                    k_4.reshape(Bqkv * Sqkv, Nk * Dqkv).to(qkv_dtype),
                    v_4.reshape(Bqkv * Sqkv, Nv * Dqkv).to(qkv_dtype)]
        else:
            return [q_out_res.to(qkv_dtype), k_cache, v_cache]
```
