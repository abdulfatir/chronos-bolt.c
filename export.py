import logging
import struct
from dataclasses import asdict
from typing import BinaryIO

import torch
import typer
from chronos import ChronosBoltPipeline
from rich.logging import RichHandler

from chronos_bolt import ChronosBolt, ChronosBoltConfig

app = typer.Typer(pretty_exceptions_enable=False)
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)


def load_hf_model(model_name: str = "autogluon/chronos-bolt-small") -> torch.nn.Module:
    pipeline = ChronosBoltPipeline.from_pretrained(model_name)
    return pipeline.inner_model


def convert_hf_model_to_pure_torch(hf_model: torch.nn.Module) -> tuple[torch.nn.Module, ChronosBoltConfig]:
    config = ChronosBoltConfig(
        d_model=hf_model.config.d_model,
        d_ff=hf_model.config.d_ff,
        d_kv=hf_model.config.d_kv,
        n_heads=hf_model.config.num_heads,
        n_encoder_layers=hf_model.config.num_layers,
        n_decoder_layers=hf_model.config.num_decoder_layers,
        rel_attn_num_buckets=hf_model.config.relative_attention_num_buckets,
        rel_attn_max_distance=hf_model.config.relative_attention_max_distance,
        layer_norm_eps=hf_model.config.layer_norm_epsilon,
        context_length=hf_model.chronos_config.context_length,
        prediction_length=hf_model.chronos_config.prediction_length,
        input_patch_size=hf_model.chronos_config.input_patch_size,
        quantiles=hf_model.chronos_config.quantiles,
    )

    model = ChronosBolt(config)
    model_state_dict = model.state_dict()
    hf_model_state_dict = hf_model.state_dict()

    def replace_prefix(old_prefix: str, new_prefix: str):
        relevant_keys = list(filter(lambda x: x.startswith(old_prefix), hf_model_state_dict.keys()))
        for key in relevant_keys:
            new_key = new_prefix + key.removeprefix(old_prefix)
            if new_key not in model_state_dict:
                logger.warning(f"Key {new_key} not found in model state dict")
            if model_state_dict[new_key].shape != hf_model_state_dict[key].shape:
                logger.warning(
                    f"Shape mismatch for key {new_key}. "
                    f"HF Model: {hf_model_state_dict[key].shape}, PyTorch Model: {model_state_dict[new_key].shape}"
                )
            model_state_dict[new_key] = hf_model_state_dict[key]

    model_state_dict["spl_tokens.weight"] = hf_model_state_dict["shared.weight"]
    replace_prefix("input_patch_embedding", "input_patch_emb")
    replace_prefix("output_patch_embedding", "output_patch_emb")

    for i in range(config.n_encoder_layers):
        replace_prefix(f"encoder.block.{i}.layer.0.layer_norm", f"encoder.layers.{i}.self_attn.layer_norm")
        replace_prefix(f"encoder.block.{i}.layer.0.SelfAttention", f"encoder.layers.{i}.self_attn.self_attn")
        replace_prefix(f"encoder.block.{i}.layer.1.DenseReluDense", f"encoder.layers.{i}.ff")
        replace_prefix(f"encoder.block.{i}.layer.1.layer_norm", f"encoder.layers.{i}.ff.layer_norm")
    model_state_dict["encoder.final_layer_norm.weight"] = hf_model_state_dict["encoder.final_layer_norm.weight"]

    for i in range(config.n_decoder_layers):
        replace_prefix(f"decoder.block.{i}.layer.0.layer_norm", f"decoder.layers.{i}.self_attn.layer_norm")
        replace_prefix(f"decoder.block.{i}.layer.0.SelfAttention", f"decoder.layers.{i}.self_attn.self_attn")
        replace_prefix(f"decoder.block.{i}.layer.1.layer_norm", f"decoder.layers.{i}.cross_attn.layer_norm")
        replace_prefix(f"decoder.block.{i}.layer.1.EncDecAttention", f"decoder.layers.{i}.cross_attn.cross_attn")
        replace_prefix(f"decoder.block.{i}.layer.2.DenseReluDense", f"decoder.layers.{i}.ff")
        replace_prefix(f"decoder.block.{i}.layer.2.layer_norm", f"decoder.layers.{i}.ff.layer_norm")
    model_state_dict["decoder.final_layer_norm.weight"] = hf_model_state_dict["decoder.final_layer_norm.weight"]

    model.load_state_dict(model_state_dict)

    return model, config


@app.command()
def hf2pt(model_name: str = "autogluon/chronos-bolt-small"):
    hf_model = load_hf_model(model_name)
    model, config = convert_hf_model_to_pure_torch(hf_model)

    logger.info(f"Num. params in HF Model: {hf_model.num_parameters()}")
    logger.info(f"Num. params in PyTorch Model: {model.num_parameters()}")

    output_path = f"{model_name.replace('/', '-')}.pt"
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, output_path)
    logger.info(f"Saved model to: {output_path}")


@torch.no_grad()
def serialize_tensor(tensor: torch.Tensor, fp: BinaryIO):
    fp.write(tensor.float().view(-1).numpy().astype("float32").tobytes())


def serialize_config(config: ChronosBoltConfig, fp: BinaryIO):
    fp.write(struct.pack("I", config.d_model))
    fp.write(struct.pack("I", config.d_ff))
    fp.write(struct.pack("I", config.d_kv))
    fp.write(struct.pack("I", config.n_heads))
    fp.write(struct.pack("I", config.n_encoder_layers))
    fp.write(struct.pack("I", config.n_decoder_layers))
    fp.write(struct.pack("I", config.rel_attn_num_buckets))
    fp.write(struct.pack("I", config.rel_attn_max_distance))
    fp.write(struct.pack("f", config.layer_norm_eps))
    fp.write(struct.pack("I", config.context_length))
    fp.write(struct.pack("I", config.prediction_length))
    fp.write(struct.pack("I", config.input_patch_size))
    fp.write(struct.pack(f"{len(config.quantiles)}f", *config.quantiles))


@app.command()
def hf2bin(model_name: str = "autogluon/chronos-bolt-small", version: int = 1):
    hf_model = load_hf_model(model_name)
    model, config = convert_hf_model_to_pure_torch(hf_model)

    logger.info(f"Num. params in HF Model: {hf_model.num_parameters()}")
    logger.info(f"Num. params in PyTorch Model: {model.num_parameters()}")

    output_path = f"{model_name.replace('/', '-')}.bin"

    with open(output_path, "wb") as fp:
        # write magic bytes as uint32 (4 bytes), 0x62306c74 is b0lt in hex
        fp.write(struct.pack("I", 0x62306C74))
        # write version as uint32 (4 bytes)
        fp.write(struct.pack("I", version))
        # write config
        serialize_config(config, fp)
        # write weights
        weights = [
            # Special tokens
            model.spl_tokens.weight,
            # Input patch embedding
            model.input_patch_emb.hidden_layer.weight,
            model.input_patch_emb.hidden_layer.bias,
            model.input_patch_emb.output_layer.weight,
            model.input_patch_emb.output_layer.bias,
            model.input_patch_emb.residual_layer.weight,
            model.input_patch_emb.residual_layer.bias,
            # Encoder
            model.encoder.layers[0].self_attn.self_attn.relative_attention_bias.weight,
            *[layer.self_attn.self_attn.q.weight for layer in model.encoder.layers],
            *[layer.self_attn.self_attn.k.weight for layer in model.encoder.layers],
            *[layer.self_attn.self_attn.v.weight for layer in model.encoder.layers],
            *[layer.self_attn.self_attn.o.weight for layer in model.encoder.layers],
            *[layer.self_attn.layer_norm.weight for layer in model.encoder.layers],
            *[layer.ff.wi.weight for layer in model.encoder.layers],
            *[layer.ff.wo.weight for layer in model.encoder.layers],
            *[layer.ff.layer_norm.weight for layer in model.encoder.layers],
            model.encoder.final_layer_norm.weight,
            # Decoder
            model.decoder.layers[0].self_attn.self_attn.relative_attention_bias.weight,
            *[layer.self_attn.self_attn.q.weight for layer in model.decoder.layers],
            *[layer.self_attn.self_attn.k.weight for layer in model.decoder.layers],
            *[layer.self_attn.self_attn.v.weight for layer in model.decoder.layers],
            *[layer.self_attn.self_attn.o.weight for layer in model.decoder.layers],
            *[layer.self_attn.layer_norm.weight for layer in model.decoder.layers],
            *[layer.cross_attn.cross_attn.q.weight for layer in model.decoder.layers],
            *[layer.cross_attn.cross_attn.k.weight for layer in model.decoder.layers],
            *[layer.cross_attn.cross_attn.v.weight for layer in model.decoder.layers],
            *[layer.cross_attn.cross_attn.o.weight for layer in model.decoder.layers],
            *[layer.cross_attn.layer_norm.weight for layer in model.decoder.layers],
            *[layer.ff.wi.weight for layer in model.decoder.layers],
            *[layer.ff.wo.weight for layer in model.decoder.layers],
            *[layer.ff.layer_norm.weight for layer in model.decoder.layers],
            model.decoder.final_layer_norm.weight,
            # Output patch embedding
            model.output_patch_emb.hidden_layer.weight,
            model.output_patch_emb.hidden_layer.bias,
            model.output_patch_emb.output_layer.weight,
            model.output_patch_emb.output_layer.bias,
            model.output_patch_emb.residual_layer.weight,
            model.output_patch_emb.residual_layer.bias,
        ]
        for weight in weights:
            serialize_tensor(weight, fp)

    logger.info(f"Saved model to: {output_path}")


if __name__ == "__main__":
    app()
