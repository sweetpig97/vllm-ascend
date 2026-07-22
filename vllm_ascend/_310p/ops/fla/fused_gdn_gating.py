import torch


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute fused GDN gating using custom op on 310P.

    Args:
        A_log: Log of A parameter, shape [num_heads]
        a: a parameter, shape [batch, num_heads]
        b: b parameter, shape [batch, num_heads]
        dt_bias: dt bias, shape [num_heads]
        beta: softplus beta parameter
        threshold: softplus threshold parameter

    Returns:
        g: gating parameter, shape [1, batch, num_heads]
        beta_output: sigmoid(b), shape [1, batch, num_heads]
    """
    A_log_typed = A_log.to(torch.float16)
    a_typed = a.to(torch.float16)
    b_typed = b.to(torch.float16)
    dt_bias_typed = dt_bias.to(torch.float16)
    return torch.ops._C_ascend.npu_fused_gdn_gating(
        A_log_typed,
        a_typed,
        b_typed,
        dt_bias_typed,
        beta,
        threshold,
    )
