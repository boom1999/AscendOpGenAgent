## 功能说明

- 算子功能：在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。
- swiglu_mode为0时的计算公式：  

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  swigluOut_i = Swiglu(dequantOut_i)=Swish(A_i)*B_i
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  其中，A<sub>i</sub>表示dequantOut<sub>i</sub>的前半部分，B<sub>i</sub>表示dequantOut<sub>i</sub>的后半部分。

- swiglu_mode为1时的计算公式：  

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  x\_glu = x\_glu.clamp(min=None, max=clamp\_limit)
  $$
  
  $$
  x\_linear = x\_linear.clamp(min=-clamp\_limit, max=clamp\_limit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(glu\_alpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + glu\_bias)
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  其中，x\_glu表示dequantOut<sub>i</sub>的偶数索引部分，x\_linear表示dequantOut<sub>i</sub>的奇数索引部分。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | shape |
|---|---|---|---|---|
| x | 输入 | 输入待处理的数据 | FLOAT16、BFLOAT16、INT32 | ND |
| weight_scale | 输入 | 权重反量化scale | FLOAT32 | ND |
| activation_scale | 输入 | 激活函数的反量化scale | FLOAT32 | ND |
| bias | 输入 | Matmul的bias | FLOAT32、FLOAT16、BFLOAT16、INT32 | ND |
| quant_scale | 输入 | 量化的scale | FLOAT32、FLOAT16 | ND |
| quant_offset | 输入 | 量化的offset | FLOAT32 | ND |
| group_index | 输入 | MoE分组需要的group_index | INT64 | ND |
| activate_left | 属性 | 是否对输入的左半部分做swiglu激活 | BOOL | - |
| quant_mode | 属性 | 量化模式（dynamic等） | STRING | - |
| dst_type | 属性 | 指定输出y的数据类型 | INT64 | - |
| round_mode | 属性 | 输出y结果的舍入模式 | STRING | - |
| activate_dim | 属性 | swish计算时选择的指定切分轴 | INT64 | - |
| swiglu_mode | 属性 | swiglu的计算模式（0=标准, 1=变体） | INT64 | - |
| clamp_limit | 属性 | 变体swiglu使用的门限值 | FLOAT32 | - |
| glu_alpha | 属性 | 变体swiglu使用的参数 | FLOAT32 | - |
| glu_bias | 属性 | 变体swiglu使用的偏差参数 | FLOAT32 | - |
| y | 输出 | 输出结果 | INT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2 | ND |
| scale | 输出 | 动态量化scale输出 | FLOAT32 | ND |

- Kirin X90/Kirin 9030：输入x不支持BFLOAT16；输入bias不支持BFLOAT16；输入quant_scale不支持FLOAT16；输出y不支持FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2。

## 约束说明

- Ascend 950PR/950DT：
  - 输入x对应activate_dim的维度需要是2的倍数，且x的维数必须大于1维。
  - 当输入x的数据类型为INT32时，weight_scale不能为空；当输入x的数据类型不为INT32时，weight_scale不允许输入。
  - 当输入x的数据类型不为INT32时，activation_scale不允许输入。
  - 当输入x的数据类型不为INT32时，bias不允许输入。
  - 当输出y的数据类型为FLOAT4_E2M1、FLOAT4_E1M2时，y的最后一维需要是2的倍数。
  - 输出y的尾轴不超过5120。
- Atlas A2/A3 训练/推理系列：
  - swiglu_mode、clamp_limit、glu_alpha和glu_bias四个参数用于GPT-OSS变体SwiGLU的使用。
  - x的最后一维需要是2的倍数，且x的维数必须大于1维。
  - 当quant_mode为static时，quant_scale和quant_offset为1维，值为1；quant_mode为dynamic时，quant_scale和quant_offset可选。
  - 算子支持的输入张量内存大小有上限：weight_scale张量内存+bias张量内存+quant_scale张量内存+quant_offset张量内存+(activation_scale张量内存+scale张量内存)/40+x张量最后一维H内存*10 < 192KB。

```python
class Model(nn.Module):
    """Dequant + Swiglu + Quant fused operator."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight_scale: torch.Tensor = None,
        activate_scale: torch.Tensor = None,
        bias: torch.Tensor = None,
        quant_scale: torch.Tensor = None,
        quant_offset: torch.Tensor = None,
        group_num: torch.Tensor = None,
        activate_left: bool = True,
        quant_mode: str = "dynamic",
        swiglu_mode: int = 0,
        clamp_limit: float = 5.0,
        glu_alpha: float = 1.0,
        glu_bias: float = 0.0,
    ) -> List[torch.Tensor]:
        """
        DequantSwigluQuant: dequant -> swiglu -> quant.

        swiglu_mode=0: out = Quant(Swish(A) * B) where A,B = split(Dequant(x))
        swiglu_mode=1: clipped swiglu with glu_alpha and glu_bias
        """
        if group_num is None:
            group_num = torch.tensor([x.shape[0]], dtype=torch.int64)

        offset = 0
        res_y = torch.zeros([x.shape[0], x.shape[1] // 2], dtype=torch.float32)
        res_scale = torch.zeros([x.shape[0]], dtype=torch.float32)

        for g_idx in range(group_num.shape[0]):
            groupIdx = group_num[g_idx].item()
            x_tensor = x[offset:offset + groupIdx].to(torch.float32)

            # Dequant
            res = x_tensor
            if weight_scale is not None:
                res = res * weight_scale[g_idx].to(torch.float32)
            if activate_scale is not None:
                res = res * activate_scale[offset:offset + groupIdx].unsqueeze(-1).to(torch.float32)
            if bias is not None:
                res = res + bias[g_idx].to(torch.float32)

            # Swiglu
            if swiglu_mode == 0:
                out = torch.chunk(res, 2, dim=-1)
                if activate_left:
                    self_tensor = out[0]
                    other = out[1]
                else:
                    self_tensor = out[1]
                    other = out[0]
                output = torch.nn.functional.silu(self_tensor) * other
            else:
                x_glu = res[..., ::2]
                x_linear = res[..., 1::2]
                x_glu = x_glu.clamp(max=clamp_limit)
                x_linear = x_linear.clamp(min=-clamp_limit, max=clamp_limit)
                out_glu = x_glu * torch.sigmoid(glu_alpha * x_glu)
                output = out_glu * (x_linear + glu_bias)

            # Quant
            if quant_scale is not None:
                output = output * quant_scale[g_idx].to(torch.float32)
            if quant_offset is not None:
                output = output + quant_offset[g_idx].to(torch.float32)

            if quant_mode == "dynamic":
                abs_val = torch.abs(output)
                max_values = torch.amax(abs_val, dim=-1)
                scale_out = max_values / 127.0
                max_values = 127.0 / max_values
                output = output * max_values.unsqueeze(1)

            output = torch.clamp(output, -128, 127)
            output = torch.round(output)
            res_y[offset:offset + groupIdx] = output
            res_scale[offset:offset + groupIdx] = scale_out
            offset = offset + groupIdx

        return [res_y.to(torch.int8), res_scale]
```
