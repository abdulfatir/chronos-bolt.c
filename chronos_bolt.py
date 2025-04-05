import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import typer

app = typer.Typer(pretty_exceptions_enable=False)


@dataclass
class ChronosBoltConfig:
    """Defaults from chronos-bolt-small"""

    d_model: int = 512
    d_ff: int = 2048
    d_kv: int = 64
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    rel_attn_num_buckets: int = 32
    rel_attn_max_distance: int = 128
    layer_norm_eps: float = 1e-6
    dropout_rate: float = 0.1  # not really used for inference
    # params specific to chronos-bolt
    context_length: int = 2048
    prediction_length: int = 64
    input_patch_size: int = 16
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


class LayerNorm(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.d_model))
        self.variance_epsilon = cfg.layer_norm_eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states


class DenseActDense(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()

        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig):
        super().__init__()
        self.wi = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout1 = nn.Dropout(cfg.dropout_rate)
        self.act = nn.ReLU()
        self.layer_norm = LayerNorm(cfg)
        self.dropout2 = nn.Dropout(cfg.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.wi(forwarded_states)
        forwarded_states = self.act(forwarded_states)
        forwarded_states = self.dropout1(forwarded_states)
        forwarded_states = self.wo(forwarded_states)
        hidden_states = hidden_states + self.dropout2(forwarded_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig, has_rel_pos_bias: bool = False, is_decoder: bool = False):
        super().__init__()
        self.d_model = cfg.d_model
        self.key_value_proj_dim = cfg.d_kv
        self.n_heads = cfg.n_heads
        self.dropout = cfg.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.is_decoder = is_decoder

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        self.has_rel_pos_bias = has_rel_pos_bias

        if self.has_rel_pos_bias:
            self.rel_attn_num_buckets = cfg.rel_attn_num_buckets
            self.rel_attn_max_distance = cfg.rel_attn_max_distance
            self.relative_attention_bias = nn.Embedding(self.rel_attn_num_buckets, self.n_heads)

    def relative_position_bucket(self, relative_position: torch.LongTensor) -> torch.LongTensor:
        bidirectional = not self.is_decoder
        num_buckets = self.rel_attn_num_buckets
        max_distance = self.rel_attn_max_distance

        relative_buckets = torch.zeros_like(relative_position)
        if bidirectional:
            # [0, 0, 0, ..., num_buckets/2, num_buckets/2, num_buckets/2 ...]
            num_buckets //= 2
            relative_buckets += torch.where(relative_position > 0, num_buckets, 0)
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.clamp(relative_position, max=0)

        # half of the buckets are for exact increments in positions
        num_exact_buckets = num_buckets // 2
        num_log_spaced_buckets = num_buckets - num_exact_buckets

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        log_ratio = torch.log(relative_position.float() / num_exact_buckets) / (
            math.log(max_distance / num_exact_buckets)
        )
        log_spaced_relative_position = num_exact_buckets + (log_ratio * num_log_spaced_buckets).to(torch.long)
        log_spaced_relative_position = torch.clamp(log_spaced_relative_position, max=num_buckets - 1)

        is_exact = relative_position < num_exact_buckets
        relative_buckets += torch.where(is_exact, relative_position, log_spaced_relative_position)

        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        device = self.relative_attention_bias.weight.device
        query_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        kv_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]

        relative_position = kv_position - query_position  # shape (query_length, key_length)
        relative_position_bucket = self.relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ):
        batch_size = hidden_states.shape[0]

        query_states = self.q(hidden_states)
        if encoder_states is None:
            # Self Attention
            key_states = self.k(hidden_states)
            value_states = self.v(hidden_states)
        else:
            # Cross Attention
            key_states = self.k(encoder_states)
            value_states = self.v(encoder_states)

        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        if position_bias is None:
            position_bias = 0.0
            if self.has_rel_pos_bias:
                position_bias = self.compute_bias(query_states.shape[2], key_states.shape[2])

        scores = torch.matmul(query_states, key_states.transpose(3, 2)) + mask + position_bias
        attn_weights = nn.functional.softmax(scores, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        return attn_output, position_bias


class SelfAttention(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig, has_rel_pos_bias: bool = False, is_decoder: bool = False):
        super().__init__()
        self.self_attn = Attention(cfg=cfg, has_rel_pos_bias=has_rel_pos_bias, is_decoder=is_decoder)
        self.layer_norm = LayerNorm(cfg=cfg)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, hidden_states, mask, position_bias):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias = self.self_attn(normed_hidden_states, mask=mask, position_bias=position_bias)
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, position_bias


class CrossAttention(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig):
        super().__init__()
        self.cross_attn = Attention(cfg=cfg, has_rel_pos_bias=False, is_decoder=True)
        self.layer_norm = LayerNorm(cfg=cfg)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, hidden_states, mask, encoder_states):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, _ = self.cross_attn(normed_hidden_states, mask=mask, encoder_states=encoder_states)
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig, has_rel_pos_bias: bool = False):
        super().__init__()
        self.self_attn = SelfAttention(cfg=cfg, has_rel_pos_bias=has_rel_pos_bias, is_decoder=False)
        self.ff = FeedForward(cfg=cfg)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor, position_bias: torch.Tensor | None):
        hidden_states, position_bias = self.self_attn(hidden_states, mask=mask, position_bias=position_bias)
        hidden_states = self.ff(hidden_states)

        return hidden_states, position_bias


class Encoder(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig):
        super().__init__()
        self.num_layers = cfg.n_encoder_layers
        self.layers = nn.ModuleList(
            [EncoderBlock(cfg=cfg, has_rel_pos_bias=(idx == 0)) for idx in range(self.num_layers)]
        )
        self.final_layer_norm = LayerNorm(cfg=cfg)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        position_bias = None
        for layer in self.layers:
            hidden_states, position_bias = layer(hidden_states, mask=mask, position_bias=position_bias)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig, has_rel_pos_bias: bool = False):
        super().__init__()
        self.self_attn = SelfAttention(cfg=cfg, has_rel_pos_bias=has_rel_pos_bias, is_decoder=True)
        self.cross_attn = CrossAttention(cfg=cfg)
        self.ff = FeedForward(cfg=cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        position_bias: torch.Tensor | None,
    ):
        hidden_states, position_bias = self.self_attn(hidden_states, mask=mask, position_bias=position_bias)
        hidden_states = self.cross_attn(hidden_states, mask=encoder_mask, encoder_states=encoder_states)
        hidden_states = self.ff(hidden_states)
        return hidden_states, position_bias


class Decoder(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig):
        super().__init__()
        self.num_layers = cfg.n_decoder_layers
        self.layers = nn.ModuleList(
            [DecoderBlock(cfg=cfg, has_rel_pos_bias=(idx == 0)) for idx in range(self.num_layers)]
        )
        self.final_layer_norm = LayerNorm(cfg=cfg)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor, encoder_states: torch.Tensor, encoder_mask: torch.Tensor
    ):
        position_bias = None
        for layer in self.layers:
            hidden_states, position_bias = layer(
                hidden_states,
                mask=mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
                position_bias=position_bias,
            )
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, out_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        return out + res


class ChronosBolt(nn.Module):
    def __init__(self, cfg: ChronosBoltConfig):
        super().__init__()
        self.cfg = cfg

        # [PAD] and [REG] tokens
        self.spl_tokens = nn.Embedding(2, cfg.d_model)

        self.input_patch_emb = PatchEmbedding(
            in_dim=cfg.input_patch_size * 2,
            h_dim=cfg.d_ff,
            out_dim=cfg.d_model,
            dropout_rate=cfg.dropout_rate,
        )
        self.encoder = Encoder(cfg=cfg)
        self.decoder = Decoder(cfg=cfg)
        self.output_patch_emb = PatchEmbedding(
            in_dim=cfg.d_model,
            h_dim=cfg.d_ff,
            out_dim=len(cfg.quantiles) * cfg.prediction_length,
            dropout_rate=cfg.dropout_rate,
        )

    def instance_norm(self, context: torch.Tensor, eps: float = 1e-5):
        loc = torch.nan_to_num(torch.nanmean(context, dim=-1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num(torch.nanmean((context - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0)
        scale = torch.where(scale == 0, eps, scale)

        return (context - loc) / scale, (loc, scale)

    def patch(self, context: torch.Tensor):
        batch_size, seq_len = context.shape
        patch_size = self.cfg.input_patch_size

        if seq_len % patch_size != 0:
            padding_size = (batch_size, patch_size - (seq_len % patch_size))
            padding = torch.full(size=padding_size, fill_value=torch.nan, dtype=context.dtype, device=context.device)
            context = torch.concat((padding, context), dim=-1)

        return context.reshape(batch_size, -1, patch_size)

    def forward(self, context: torch.Tensor):
        batch_size, seq_len = context.shape

        # Standardize
        context, (loc, scale) = self.instance_norm(context)

        # Patch
        patched_context = self.patch(context)
        nan_elements = torch.isnan(patched_context)
        patched_context = torch.nan_to_num(patched_context, nan=0.0)
        patched_mask = nan_elements.logical_not()
        patched_context = torch.cat([patched_context, patched_mask], dim=-1)

        # Construct encoder inputs
        input_emb = self.input_patch_emb(patched_context)
        reg_token_emb = self.spl_tokens(torch.ones((batch_size, 1), dtype=torch.long, device=input_emb.device))
        input_emb = torch.cat([input_emb, reg_token_emb], dim=-2)

        # Construct attention mask
        attention_mask = patched_mask.sum(dim=-1) > 0
        attention_mask = torch.cat(
            [attention_mask, torch.ones(batch_size, 1, dtype=bool, device=attention_mask.device)], dim=-1
        )
        attention_mask = torch.where(attention_mask, 0.0, float("-inf"))  # inverted mask
        attention_mask = attention_mask[:, None, None, :]

        # Pass through encoder
        encoder_states = self.encoder(hidden_states=input_emb, mask=attention_mask)

        # Pass through decoder
        decoder_input_emb = self.spl_tokens(torch.zeros((batch_size, 1), dtype=torch.long, device=input_emb.device))
        decoder_mask = torch.zeros(batch_size, 1, dtype=bool, device=input_emb.device)  # inverted mask
        decoder_mask = decoder_mask[:, None, None, :]
        decoder_states = self.decoder(
            decoder_input_emb, mask=decoder_mask, encoder_states=encoder_states, encoder_mask=attention_mask
        )
        decoder_states = decoder_states.squeeze(-2)

        # Map decoder embedding to quantile forecast
        output_shape = (
            batch_size,
            len(self.cfg.quantiles),
            self.cfg.prediction_length,
        )
        quantile_forecast = self.output_patch_emb(decoder_states)
        quantile_forecast = quantile_forecast * scale + loc

        return quantile_forecast.reshape(*output_shape)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pad_and_stack(context: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(len(c) for c in context)

    context = [torch.cat((torch.full((max_len - len(c),), torch.nan), c), dim=0) for c in context]
    return torch.stack(context)


@app.command()
def main(model_ckpt_path: Path, past_data_path: Path, forecast_path: Path, device: str = "cpu"):
    ckpt = torch.load(model_ckpt_path, map_location="cpu")
    config = ChronosBoltConfig(**ckpt["config"])
    model = ChronosBolt(cfg=config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    past_df = pd.read_csv(past_data_path)
    item_ids = []
    context = []
    for item_id, time_series in past_df.groupby("item_id"):
        item_ids.append(item_id)
        context.append(torch.tensor(time_series["target"].values, dtype=torch.float32))
    context = pad_and_stack(context).to(device)

    with torch.inference_mode():
        out = model(context).cpu().numpy()

    forecast_dfs = []
    for item_id, forecast in zip(item_ids, out):
        forecast_df = pd.DataFrame(
            {"item_id": [item_id] * forecast.shape[1], **{f"q{q:.1f}": v for q, v in zip(config.quantiles, forecast)}}
        )
        forecast_dfs.append(forecast_df)
    pd.concat(forecast_dfs).to_csv(forecast_path, index=False, float_format="%.4f")


if __name__ == "__main__":
    app()
