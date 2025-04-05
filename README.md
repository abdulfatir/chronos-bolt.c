# chronos-bolt.c

A pure C implementation of the Chronos-Bolt pretrained time series forecasting model. Inspired by the [llama2.c](https://github.com/karpathy/llama2.c) project from Andrej Karpathy.

> [!NOTE]
> This is a weekend side project I wrote for fun and learning, outside of my work at AWS. As such, the code in this repository is provided "as-is". This is neither the official nor the recommended way to use Chronos-Bolt for practical use cases. For that, please refer to the official [chronos repository](https://github.com/amazon-science/chronos-forecasting) or [AutoGluon](https://auto.gluon.ai/stable/tutorials/timeseries/index.html).

Apart from `chronos_bolt.c`, this repo also includes a reference pure PyTorch implementation of Chronos-Bolt inside `chronos_bolt.py`.

## Usage

- Create a new python enviroment and install pip requirements [using uv](https://docs.astral.sh/uv/getting-started/installation/).
```sh
# Remove uv from the beginning, if you don't have uv installed
uv pip install -r requirements.txt
```

### Building and running the C version

- Export the model checkpoint in a custom binary format used by `chronos_bolt.c`. The following command will generate `autogluon-chronos-bolt-small.bin`.
```sh
python export.py hf2bin --model-name autogluon/chronos-bolt-small
```
- Build the project. The following command will generate a `chronos_bolt` executable.
```sh
make
```
- Run `chronos_bolt` on sample `data.csv`. The forecasts will be saved to `forecast-c.csv`. **NOTE**: The C implementation does NOT handle missing values.
```sh
./chronos_bolt autogluon-chronos-bolt-small.bin data.csv forecast-c.csv
```

### Running the reference pure PyTorch version

- Export the model checkpoint in a PyTorch format used by `chronos_bolt.py`. The following command will generate `autogluon-chronos-bolt-small.pt`.
```sh
python export.py hf2pt --model-name autogluon/chronos-bolt-small
```
- Run `chronos_bolt.py` on sample `data.csv`. The forecasts will be saved to `forecast-pt.csv`.
```sh
python chronos_bolt.py autogluon-chronos-bolt-small.pt data.csv forecast-pt.csv
```

## Other Details

### Data Format

The CSV file should be in the following format, where the `item_{i}` is a string denoting a unique time series and `value_{j}` are uniformly spaced numeric values. The `item_id`s and values should be in the correct order as the C version does not perform any sorting or grouping. Missing values are not handled by the C version.

```csv
item_id,target
item_1,value_1
item_1,value_2
item_1,value_3
item_1,value_4
...
item_2,value_1
item_2,value_2
item_2,value_3
...
```

### Custom Binary Checkpoint Format

When running `python export.py hf2bin --model-name <hf-model-id>`, the model is exported in a custom binary format. Here's how the binary file is organized.

#### 1. Header
The file begins with two 4-byte unsigned integers:
- **Magic Bytes**: `0x62306C74` (`'b0lt'` in hex), used to identify the file format.
- **Version**: Format version number (currently `1`).

#### 2. Model Configuration
The model configuration is written in this exact order:

| Field                     | Type     |
|--------------------------|----------|
| `d_model`                | `uint32` |
| `d_ff`                   | `uint32` |
| `d_kv`                   | `uint32` |
| `n_heads`                | `uint32` |
| `n_encoder_layers`       | `uint32` |
| `n_decoder_layers`       | `uint32` |
| `rel_attn_num_buckets`   | `uint32` |
| `rel_attn_max_distance`  | `uint32` |
| `layer_norm_eps`         | `float32` |
| `context_length`         | `uint32` |
| `prediction_length`      | `uint32` |
| `input_patch_size`       | `uint32` |
| `quantiles`              | `float32[9]` |

#### 3. Model Weights
The model's weights are written in a fixed order as 1-d float arrays.

##### Special Tokens
- `spl_tokens.weight`

##### Input Patch Embedding
- `hidden_layer.weight`
- `hidden_layer.bias`
- `output_layer.weight`
- `output_layer.bias`
- `residual_layer.weight`
- `residual_layer.bias`

##### Encoder
- `relative_attention_bias.weight`
- For each encoder layer:
  - `self_attn.q/k/v/o.weight`
  - `self_attn.layer_norm.weight`
  - `ff.wi/wo.weight`
  - `ff.layer_norm.weight`
- `final_layer_norm.weight`

##### Decoder
- `relative_attention_bias.weight`
- For each decoder layer:
  - `self_attn.q/k/v/o.weight`
  - `self_attn.layer_norm.weight`
  - `cross_attn.q/k/v/o.weight`
  - `cross_attn.layer_norm.weight`
  - `ff.wi/wo.weight`
  - `ff.layer_norm.weight`
- `final_layer_norm.weight`

##### Output Patch Embedding
- `hidden_layer.weight`
- `hidden_layer.bias`
- `output_layer.weight`
- `output_layer.bias`
- `residual_layer.weight`
- `residual_layer.bias`
