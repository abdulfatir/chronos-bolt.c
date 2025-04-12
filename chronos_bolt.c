#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define NUM_QUANTILES 9
#define CHECKPOINT_MAGIC 0x62306C74 // hex for "b0lt"

// Structs
typedef struct {
    uint32_t d_model;
    uint32_t d_ff;
    uint32_t d_kv;
    uint32_t n_heads;
    uint32_t n_encoder_layers;
    uint32_t n_decoder_layers;
    uint32_t rel_attn_num_buckets;
    uint32_t rel_attn_max_distance;
    float layer_norm_eps;
    uint32_t context_length;
    uint32_t prediction_length;
    uint32_t input_patch_size;
    float quantiles[NUM_QUANTILES];
} ChronosBoltConfig;

typedef struct {
    // Special Tokens
    float *spl_tokens; // (2, d_model)
    // Input Patch Embedding
    float *inp_patch_emb_hid_w; // (d_ff, input_patch_size * 2)
    float *inp_patch_emb_hid_b; // (d_ff, )
    float *inp_patch_emb_out_w; // (d_model, d_ff)
    float *inp_patch_emb_out_b; // (d_model, )
    float *inp_patch_emb_res_w; // (d_model, input_patch_size * 2)
    float *inp_patch_emb_res_b; // (d_model, )
    // Encoder
    float *encoder_rel_pos_bias; // (num_buckets, n_heads)
    float *encoder_self_attn_q;  // (n_encoder_layers, n_heads * d_kv, d_model)
    float *encoder_self_attn_k;  // (n_encoder_layers, n_heads * d_kv, d_model)
    float *encoder_self_attn_v;  // (n_encoder_layers, n_heads * d_kv, d_model)
    float *encoder_self_attn_o;  // (n_encoder_layers, d_model, n_heads * d_kv)
    float *encoder_self_attn_ln; // (n_encoder_layers, d_model)
    float *encoder_ff_wi;        // (n_encoder_layers, d_ff, d_model)
    float *encoder_ff_wo;        // (n_encoder_layers, d_model, d_ff)
    float *encoder_ff_ln;        // (n_encoder_layers, d_model)
    float *encoder_final_ln;     // (d_model,)
    // Decoder
    float *decoder_rel_pos_bias;  // (num_buckets, n_heads)
    float *decoder_self_attn_q;   // (n_decoder_layers, n_heads * d_kv, d_model)
    float *decoder_self_attn_k;   // (n_decoder_layers, n_heads * d_kv, d_model)
    float *decoder_self_attn_v;   // (n_decoder_layers, n_heads * d_kv, d_model)
    float *decoder_self_attn_o;   // (n_decoder_layers, d_model, n_heads * d_kv)
    float *decoder_self_attn_ln;  // (n_decoder_layers, d_model)
    float *decoder_cross_attn_q;  // (n_decoder_layers, n_heads * d_kv, d_model)
    float *decoder_cross_attn_k;  // (n_decoder_layers, n_heads * d_kv, d_model)
    float *decoder_cross_attn_v;  // (n_decoder_layers, n_heads * d_kv, d_model)
    float *decoder_cross_attn_o;  // (n_decoder_layers, d_model, n_heads * d_kv)
    float *decoder_cross_attn_ln; // (n_decoder_layers, d_model)
    float *decoder_ff_wi;         // (n_decoder_layers, d_ff, d_model)
    float *decoder_ff_wo;         // (n_decoder_layers, d_model, d_ff)
    float *decoder_ff_ln;         // (n_decoder_layers, d_model)
    float *decoder_final_ln;      // (d_model,)
    // Output Patch Embedding
    float *out_patch_emb_hid_w; // (d_ff, d_model)
    float *out_patch_emb_hid_b; // (d_ff,)
    float *out_patch_emb_out_w; // (NUM_QUANTILES * prediction_length, d_ff)
    float *out_patch_emb_out_b; // (NUM_QUANTILES * prediction_length,)
    float *out_patch_emb_res_w; // (NUM_QUANTILES * prediction_length, d_model)
    float *out_patch_emb_res_b; // (NUM_QUANTILES * prediction_length,)
} ChronosBoltWeights;

typedef struct {
    float *embeds;       // Encoder state (num_patches + 1, d_model)
    float *buf_d;        // Another buffer for encoder state for residual (num_patches + 1, d_model)
    float *buf_h;        // Buffer for feed forward for residual (num_patches + 1, d_ff)
    float *enc_pos_bias; // Encoder position bias (num_patches + 1, n_heads, num_patches + 1)
    float *dec_embeds;   // Decoder state (1, d_model)
    float *dec_buf_d;    // Another buffer for decoder state for residual (1, d_model)
    float *dec_buf_h;    // Buffer for feed forward for residual (1, d_ff)
    float *dec_pos_bias; // Decoder position bias (1, n_heads, 1)
} RunState;

typedef struct {
    float *query;    // Buffer for query (q_len, n_heads, d_kv)
    float *key;      // Buffer for key (kv_len, n_heads, d_kv)
    float *value;    // Buffer for value (kv_len, n_heads, d_kv)
    float *score;    // Buffer for scores (q_len, n_heads, kv_len)
    float *attn_out; // Buffer for attention output (q_len, n_heads, d_kv)
} AttentionState;

typedef struct {
    ChronosBoltConfig config;
    ChronosBoltWeights ws;
    RunState run_state;
} ChronosBolt;

typedef struct {
    float *target;
    int size;
    char *item_id;
} TimeSeries;

// Checkpoint utilities

void malloc_weights(ChronosBoltWeights *ws, ChronosBoltConfig *config) {
    const int input_patch_size = config->input_patch_size;
    const int d_model = config->d_model;
    const int d_ff = config->d_ff;
    const int n_heads = config->n_heads;
    const int d_kv = config->d_kv;
    const int rel_attn_num_buckets = config->rel_attn_num_buckets;
    const int n_encoder_layers = config->n_encoder_layers;
    const int n_decoder_layers = config->n_decoder_layers;
    const int prediction_length = config->prediction_length;

    ws->spl_tokens = (float *)malloc(2 * d_model * sizeof(float));
    ws->inp_patch_emb_hid_w = (float *)malloc(d_ff * input_patch_size * 2 * sizeof(float));
    ws->inp_patch_emb_hid_b = (float *)malloc(d_ff * sizeof(float));
    ws->inp_patch_emb_out_w = (float *)malloc(d_model * d_ff * sizeof(float));
    ws->inp_patch_emb_out_b = (float *)malloc(d_model * sizeof(float));
    ws->inp_patch_emb_res_w = (float *)malloc(d_model * input_patch_size * 2 * sizeof(float));
    ws->inp_patch_emb_res_b = (float *)malloc(d_model * sizeof(float));
    ws->encoder_rel_pos_bias = (float *)malloc(rel_attn_num_buckets * n_heads * sizeof(float));
    ws->encoder_self_attn_q = (float *)malloc(n_encoder_layers * n_heads * d_kv * d_model * sizeof(float));
    ws->encoder_self_attn_k = (float *)malloc(n_encoder_layers * n_heads * d_kv * d_model * sizeof(float));
    ws->encoder_self_attn_v = (float *)malloc(n_encoder_layers * n_heads * d_kv * d_model * sizeof(float));
    ws->encoder_self_attn_o = (float *)malloc(n_encoder_layers * d_model * n_heads * d_kv * sizeof(float));
    ws->encoder_self_attn_ln = (float *)malloc(n_encoder_layers * d_model * sizeof(float));
    ws->encoder_ff_wi = (float *)malloc(n_encoder_layers * d_ff * d_model * sizeof(float));
    ws->encoder_ff_wo = (float *)malloc(n_encoder_layers * d_model * d_ff * sizeof(float));
    ws->encoder_ff_ln = (float *)malloc(n_encoder_layers * d_model * sizeof(float));
    ws->encoder_final_ln = (float *)malloc(d_model * sizeof(float));
    ws->decoder_rel_pos_bias = (float *)malloc(rel_attn_num_buckets * n_heads * sizeof(float));
    ws->decoder_self_attn_q = (float *)malloc(n_decoder_layers * n_heads * d_kv * d_model * sizeof(float));
    ws->decoder_self_attn_k = (float *)malloc(n_decoder_layers * n_heads * d_kv * d_model * sizeof(float));
    ws->decoder_self_attn_v = (float *)malloc(n_decoder_layers * n_heads * d_kv * d_model * sizeof(float));
    ws->decoder_self_attn_o = (float *)malloc(n_decoder_layers * d_model * n_heads * d_kv * sizeof(float));
    ws->decoder_self_attn_ln = (float *)malloc(n_decoder_layers * d_model * sizeof(float));
    ws->decoder_cross_attn_q = (float *)malloc(n_decoder_layers * n_heads * d_kv * d_model * sizeof(float));
    ws->decoder_cross_attn_k = (float *)malloc(n_decoder_layers * n_heads * d_kv * d_model * sizeof(float));
    ws->decoder_cross_attn_v = (float *)malloc(n_decoder_layers * n_heads * d_kv * d_model * sizeof(float));
    ws->decoder_cross_attn_o = (float *)malloc(n_decoder_layers * d_model * n_heads * d_kv * sizeof(float));
    ws->decoder_cross_attn_ln = (float *)malloc(n_decoder_layers * d_model * sizeof(float));
    ws->decoder_ff_wi = (float *)malloc(n_decoder_layers * d_ff * d_model * sizeof(float));
    ws->decoder_ff_wo = (float *)malloc(n_decoder_layers * d_model * d_ff * sizeof(float));
    ws->decoder_ff_ln = (float *)malloc(n_decoder_layers * d_model * sizeof(float));
    ws->decoder_final_ln = (float *)malloc(d_model * sizeof(float));
    ws->out_patch_emb_hid_w = (float *)malloc(d_ff * d_model * sizeof(float));
    ws->out_patch_emb_hid_b = (float *)malloc(d_ff * sizeof(float));
    ws->out_patch_emb_out_w = (float *)malloc(NUM_QUANTILES * prediction_length * d_ff * sizeof(float));
    ws->out_patch_emb_out_b = (float *)malloc(NUM_QUANTILES * prediction_length * sizeof(float));
    ws->out_patch_emb_res_w = (float *)malloc(NUM_QUANTILES * prediction_length * d_model * sizeof(float));
    ws->out_patch_emb_res_b = (float *)malloc(NUM_QUANTILES * prediction_length * sizeof(float));

    if (ws->spl_tokens == NULL || ws->inp_patch_emb_hid_w == NULL || ws->inp_patch_emb_hid_b == NULL ||
        ws->inp_patch_emb_out_w == NULL || ws->inp_patch_emb_out_b == NULL || ws->inp_patch_emb_res_w == NULL ||
        ws->inp_patch_emb_res_b == NULL || ws->encoder_rel_pos_bias == NULL || ws->encoder_self_attn_q == NULL ||
        ws->encoder_self_attn_k == NULL || ws->encoder_self_attn_v == NULL || ws->encoder_self_attn_o == NULL ||
        ws->encoder_self_attn_ln == NULL || ws->encoder_ff_wi == NULL || ws->encoder_ff_wo == NULL ||
        ws->encoder_ff_ln == NULL || ws->encoder_final_ln == NULL || ws->decoder_rel_pos_bias == NULL ||
        ws->decoder_self_attn_q == NULL || ws->decoder_self_attn_k == NULL || ws->decoder_self_attn_v == NULL ||
        ws->decoder_self_attn_o == NULL || ws->decoder_self_attn_ln == NULL || ws->decoder_cross_attn_q == NULL ||
        ws->decoder_cross_attn_k == NULL || ws->decoder_cross_attn_v == NULL || ws->decoder_cross_attn_o == NULL ||
        ws->decoder_cross_attn_ln == NULL || ws->decoder_ff_wi == NULL || ws->decoder_ff_wo == NULL ||
        ws->decoder_ff_ln == NULL || ws->decoder_final_ln == NULL || ws->out_patch_emb_hid_w == NULL ||
        ws->out_patch_emb_hid_b == NULL || ws->out_patch_emb_out_w == NULL || ws->out_patch_emb_out_b == NULL ||
        ws->out_patch_emb_res_w == NULL || ws->out_patch_emb_res_b == NULL) {
        fprintf(stderr, "Error: could not allocate memory for weights\n");
        exit(EXIT_FAILURE);
    }
}

void read_weights(FILE *fp, ChronosBoltWeights *ws, ChronosBoltConfig *config) {
    const int input_patch_size = config->input_patch_size;
    const int d_model = config->d_model;
    const int d_ff = config->d_ff;
    const int n_heads = config->n_heads;
    const int d_kv = config->d_kv;
    const int rel_attn_num_buckets = config->rel_attn_num_buckets;
    const int n_encoder_layers = config->n_encoder_layers;
    const int n_decoder_layers = config->n_decoder_layers;
    const int prediction_length = config->prediction_length;

    fread(ws->spl_tokens, 2 * d_model * sizeof(float), 1, fp);
    fread(ws->inp_patch_emb_hid_w, d_ff * input_patch_size * 2 * sizeof(float), 1, fp);
    fread(ws->inp_patch_emb_hid_b, d_ff * sizeof(float), 1, fp);
    fread(ws->inp_patch_emb_out_w, d_model * d_ff * sizeof(float), 1, fp);
    fread(ws->inp_patch_emb_out_b, d_model * sizeof(float), 1, fp);
    fread(ws->inp_patch_emb_res_w, d_model * input_patch_size * 2 * sizeof(float), 1, fp);
    fread(ws->inp_patch_emb_res_b, d_model * sizeof(float), 1, fp);
    fread(ws->encoder_rel_pos_bias, rel_attn_num_buckets * n_heads * sizeof(float), 1, fp);
    fread(ws->encoder_self_attn_q, n_encoder_layers * n_heads * d_kv * d_model * sizeof(float), 1, fp);
    fread(ws->encoder_self_attn_k, n_encoder_layers * n_heads * d_kv * d_model * sizeof(float), 1, fp);
    fread(ws->encoder_self_attn_v, n_encoder_layers * n_heads * d_kv * d_model * sizeof(float), 1, fp);
    fread(ws->encoder_self_attn_o, n_encoder_layers * d_model * n_heads * d_kv * sizeof(float), 1, fp);
    fread(ws->encoder_self_attn_ln, n_encoder_layers * d_model * sizeof(float), 1, fp);
    fread(ws->encoder_ff_wi, n_encoder_layers * d_ff * d_model * sizeof(float), 1, fp);
    fread(ws->encoder_ff_wo, n_encoder_layers * d_model * d_ff * sizeof(float), 1, fp);
    fread(ws->encoder_ff_ln, n_encoder_layers * d_model * sizeof(float), 1, fp);
    fread(ws->encoder_final_ln, d_model * sizeof(float), 1, fp);
    fread(ws->decoder_rel_pos_bias, rel_attn_num_buckets * n_heads * sizeof(float), 1, fp);
    fread(ws->decoder_self_attn_q, n_decoder_layers * n_heads * d_kv * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_self_attn_k, n_decoder_layers * n_heads * d_kv * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_self_attn_v, n_decoder_layers * n_heads * d_kv * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_self_attn_o, n_decoder_layers * d_model * n_heads * d_kv * sizeof(float), 1, fp);
    fread(ws->decoder_self_attn_ln, n_decoder_layers * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_cross_attn_q, n_decoder_layers * n_heads * d_kv * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_cross_attn_k, n_decoder_layers * n_heads * d_kv * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_cross_attn_v, n_decoder_layers * n_heads * d_kv * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_cross_attn_o, n_decoder_layers * d_model * n_heads * d_kv * sizeof(float), 1, fp);
    fread(ws->decoder_cross_attn_ln, n_decoder_layers * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_ff_wi, n_decoder_layers * d_ff * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_ff_wo, n_decoder_layers * d_model * d_ff * sizeof(float), 1, fp);
    fread(ws->decoder_ff_ln, n_decoder_layers * d_model * sizeof(float), 1, fp);
    fread(ws->decoder_final_ln, d_model * sizeof(float), 1, fp);
    fread(ws->out_patch_emb_hid_w, d_ff * d_model * sizeof(float), 1, fp);
    fread(ws->out_patch_emb_hid_b, d_ff * sizeof(float), 1, fp);
    fread(ws->out_patch_emb_out_w, NUM_QUANTILES * prediction_length * d_ff * sizeof(float), 1, fp);
    fread(ws->out_patch_emb_out_b, NUM_QUANTILES * prediction_length * sizeof(float), 1, fp);
    fread(ws->out_patch_emb_res_w, NUM_QUANTILES * prediction_length * d_model * sizeof(float), 1, fp);
    fread(ws->out_patch_emb_res_b, NUM_QUANTILES * prediction_length * sizeof(float), 1, fp);
}

void free_weights(ChronosBoltWeights *ws) {
    free(ws->spl_tokens);
    free(ws->inp_patch_emb_hid_w);
    free(ws->inp_patch_emb_hid_b);
    free(ws->inp_patch_emb_out_w);
    free(ws->inp_patch_emb_out_b);
    free(ws->inp_patch_emb_res_w);
    free(ws->inp_patch_emb_res_b);
    free(ws->encoder_rel_pos_bias);
    free(ws->encoder_self_attn_q);
    free(ws->encoder_self_attn_k);
    free(ws->encoder_self_attn_v);
    free(ws->encoder_self_attn_o);
    free(ws->encoder_self_attn_ln);
    free(ws->encoder_ff_wi);
    free(ws->encoder_ff_wo);
    free(ws->encoder_ff_ln);
    free(ws->encoder_final_ln);
    free(ws->decoder_rel_pos_bias);
    free(ws->decoder_self_attn_q);
    free(ws->decoder_self_attn_k);
    free(ws->decoder_self_attn_v);
    free(ws->decoder_self_attn_o);
    free(ws->decoder_self_attn_ln);
    free(ws->decoder_cross_attn_q);
    free(ws->decoder_cross_attn_k);
    free(ws->decoder_cross_attn_v);
    free(ws->decoder_cross_attn_o);
    free(ws->decoder_cross_attn_ln);
    free(ws->decoder_ff_wi);
    free(ws->decoder_ff_wo);
    free(ws->decoder_ff_ln);
    free(ws->decoder_final_ln);
    free(ws->out_patch_emb_hid_w);
    free(ws->out_patch_emb_hid_b);
    free(ws->out_patch_emb_out_w);
    free(ws->out_patch_emb_out_b);
    free(ws->out_patch_emb_res_w);
    free(ws->out_patch_emb_res_b);
}

void load_checkpoint(char *ckpt_path, ChronosBolt *chronos_bolt) {
    printf("Loading checkpoint: %s\n", ckpt_path);

    FILE *fp = fopen(ckpt_path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: could not open checkpoint file %s\n", ckpt_path);
        exit(EXIT_FAILURE);
    }
    // Read magic number
    uint32_t magic_number;
    fread(&magic_number, sizeof(uint32_t), 1, fp);
    if (magic_number != CHECKPOINT_MAGIC) {
        fprintf(stderr, "Error: invalid checkpoint file %s\n", ckpt_path);
        exit(EXIT_FAILURE);
    }

    // Read version
    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, fp);

    // Read config
    fread(&chronos_bolt->config, sizeof(ChronosBoltConfig), 1, fp);

    // Read weights
    ChronosBoltWeights *ws = &chronos_bolt->ws;
    malloc_weights(ws, &chronos_bolt->config);
    read_weights(fp, ws, &chronos_bolt->config);

    int byte = fgetc(fp);
    if (byte != EOF) {
        fprintf(stderr, "Error: ckpt has data beyond what's expected!\n");
        exit(EXIT_FAILURE);
    }

    fclose(fp);
}

// Utilities to read from csv

int read_csv(const char *csv_path, TimeSeries **ts_array) {
    printf("Reading csv file: %s\n", csv_path);

    FILE *file = fopen(csv_path, "r");
    if (file == NULL) {
        perror("Error opening csv file");
        return -1;
    }

    char line[MAX_LINE_LENGTH];
    int num_series = 0;
    char prev_item_id[100] = "";
    TimeSeries *current_series = NULL;

    // skip header
    fgets(line, sizeof(line), file);
    char col0[100];
    char col1[100];

    if (sscanf(line, "%99[^,],%99s", col0, col1) != 2) {
        fprintf(stderr, "Error parsing header line: %s\n", line);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    if (strcmp(col0, "item_id") != 0 || strcmp(col1, "target") != 0) {
        fprintf(stderr, "Error: csv header does not match expected format item_id,target\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    while (fgets(line, sizeof(line), file)) {
        float target;
        char item_id[1024];

        // Parse the item_id and target
        if (sscanf(line, "%99[^,],%f", item_id, &target) != 2) {
            fprintf(stderr, "Error parsing line: %s\n", line);
            fclose(file);
            return -1;
        }

        // Check if we need to start a new TimeSeries
        if (strcmp(item_id, prev_item_id) != 0) {
            // Expand memory for ts_array by 1 x TimeSeries
            *ts_array = realloc(*ts_array, (num_series + 1) * sizeof(TimeSeries));
            current_series = &(*ts_array)[num_series];

            // Initialize the new TimeSeries struct
            current_series->target = malloc(sizeof(float));
            current_series->target[0] = target;
            current_series->size = 1;
            current_series->item_id = malloc(strlen(item_id) + 1);
            strcpy(current_series->item_id, item_id);

            strcpy(prev_item_id, item_id);
            num_series++;
        } else {
            current_series->size++;
            current_series->target = realloc(current_series->target, current_series->size * sizeof(float));
            current_series->target[current_series->size - 1] = target;
        }
    }
    fclose(file);
    return num_series;
}

void free_time_series(TimeSeries *time_series_array, int count) {
    for (int i = 0; i < count; i++) {
        free(time_series_array[i].target);
        free(time_series_array[i].item_id);
    }
    free(time_series_array);
}

// Alloc and dealloc for run state

void malloc_run_state(const int num_patches, RunState *run_state, ChronosBoltConfig *config) {
    const int d_model = config->d_model;
    const int d_ff = config->d_ff;
    const int n_heads = config->n_heads;

    run_state->embeds = (float *)malloc((num_patches + 1) * d_model * sizeof(float));
    run_state->buf_d = (float *)malloc((num_patches + 1) * d_model * sizeof(float));
    run_state->buf_h = (float *)malloc((num_patches + 1) * d_ff * sizeof(float));
    run_state->enc_pos_bias = (float *)malloc((num_patches + 1) * n_heads * (num_patches + 1) * sizeof(float));
    run_state->dec_embeds = (float *)malloc(d_model * sizeof(float));
    run_state->dec_buf_d = (float *)malloc(d_model * sizeof(float));
    run_state->dec_buf_h = (float *)malloc(d_ff * sizeof(float));
    run_state->dec_pos_bias = (float *)malloc(n_heads * sizeof(float));

    if (run_state->embeds == NULL || run_state->buf_d == NULL || run_state->buf_h == NULL ||
        run_state->enc_pos_bias == NULL || run_state->dec_embeds == NULL || run_state->dec_buf_d == NULL ||
        run_state->dec_buf_h == NULL || run_state->dec_pos_bias == NULL) {
        fprintf(stderr, "Error: could not allocate memory for run state\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState *run_state) {
    free(run_state->embeds);
    free(run_state->buf_d);
    free(run_state->buf_h);
    free(run_state->enc_pos_bias);
    free(run_state->dec_embeds);
    free(run_state->dec_buf_d);
    free(run_state->dec_buf_h);
    free(run_state->dec_pos_bias);
}

void malloc_attn_state(int q_len, int kv_len, AttentionState *attn_state, ChronosBoltConfig *config) {
    const int d_kv = config->d_kv;
    const int n_heads = config->n_heads;

    attn_state->query = (float *)malloc(q_len * n_heads * d_kv * sizeof(float));
    attn_state->key = (float *)malloc(kv_len * n_heads * d_kv * sizeof(float));
    attn_state->value = (float *)malloc(kv_len * n_heads * d_kv * sizeof(float));
    attn_state->score = (float *)malloc(q_len * n_heads * kv_len * sizeof(float));
    attn_state->attn_out = (float *)malloc(q_len * n_heads * d_kv * sizeof(float));

    if (attn_state->query == NULL || attn_state->key == NULL || attn_state->value == NULL ||
        attn_state->score == NULL || attn_state->attn_out == NULL) {
        fprintf(stderr, "Error: could not allocate memory for attention state\n");
        exit(EXIT_FAILURE);
    }
}

void free_attn_state(AttentionState *attn_state) {
    free(attn_state->query);
    free(attn_state->key);
    free(attn_state->value);
    free(attn_state->score);
    free(attn_state->attn_out);
}

// Ops

void add(float *out, float *x, float *y, int d) {
    // Element-wise add
    // x: (d,)
    // y: (d,)
    // out: (d,) = x + y

    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        out[i] = x[i] + y[i];
    }
}

void matmul(float *out, float *w, float *x, int d, int n) {
    // Matrix multiplication
    // w: (d, n)
    // x: (n,)
    // out: (d,) = w @ x

    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float tmp = 0.0f;
        for (int j = 0; j < n; j++) {
            tmp += w[i * n + j] * x[j];
        }
        out[i] = tmp;
    }
}

void batched_matmul(float *out, float *w, float *x, int d, int n, int m) {
    // Batched version of rms_norm above
    // m: batch size

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        matmul(out + i * d, w, x + i * n, d, n);
    }
}

void matmul_with_bias(float *out, float *w, float *b, float *x, int d, int n) {
    // Matrix multiplication with bias
    // w: (d, n)
    // b: (d,)
    // x: (n,)
    // out: (d,) = w @ x + b

    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float tmp = 0.0f;
        for (int j = 0; j < n; j++) {
            tmp += w[i * n + j] * x[j];
        }
        out[i] = tmp + b[i];
    }
}

void rmsnorm(float *out, float *x, float *weight, int d, float eps) {
    // Root mean square normalization
    // x: (d,)
    // weight: (d,)
    // out: (d,) = x * weight / sqrt(sum(x^2) / d + eps)

    float ss = 0.0f;

    #pragma omp parallel for reduction(+ : ss)
    for (int j = 0; j < d; j++) {
        ss += x[j] * x[j];
    }
    ss /= d;
    ss += eps;
    ss = 1.0f / sqrtf(ss);

    #pragma omp parallel for
    for (int j = 0; j < d; j++) {
        out[j] = x[j] * ss * weight[j];
    }
}

void batched_rmsnorm(float *out, float *x, float *weight, int d, int n, float eps) {
    // Batched version of rms_norm above
    // n: batch size

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        rmsnorm(out + i * d, x + i * d, weight, d, eps);
    }
}

void softmax(float *x, int d) {
    // Softmax
    // x (d,)
    // out (d,) = exp(x) / sum(exp(x))

    float max_val = -FLT_MAX;

    #pragma omp parallel for reduction(max : max_val)
    for (int i = 0; i < d; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < d; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        x[i] /= sum;
    }
}

void t5_relative_pos_bias(float *out, float *pos_bias_w, int q_len, int kv_len, bool bidirectional,
                          ChronosBoltConfig *config) {
    // out (q_len, n_heads, kv_len)

    int num_buckets = config->rel_attn_num_buckets;
    int max_distance = config->rel_attn_max_distance;
    int n_heads = config->n_heads;

    if (bidirectional) {
        num_buckets /= 2;
    }

    int n_exact = num_buckets / 2;
    int n_log_sp = num_buckets - n_exact;

    int *rel_pos = (int *)calloc(q_len * kv_len, sizeof(int));
    int *rel_bkt = (int *)calloc(q_len * kv_len, sizeof(int));
    for (int i = 0; i < q_len; i++) {
        for (int j = 0; j < kv_len; j++) {
            if (bidirectional) {
                rel_bkt[i * kv_len + j] = j > i ? num_buckets : 0;
                rel_pos[i * kv_len + j] = abs(j - i);
            } else {
                rel_pos[i * kv_len + j] = -(j - i > 0) ? 0 : j - i;
            }

            float log_ratio =
                log((float)rel_pos[i * kv_len + j] / (float)n_exact) / log((float)max_distance / (float)n_exact);
            int log_sp_pos = n_exact + (int)(log_ratio * n_log_sp);
            log_sp_pos = log_sp_pos > num_buckets ? num_buckets : log_sp_pos;
            rel_bkt[i * kv_len + j] += (rel_pos[i * kv_len + j] < n_exact ? rel_pos[i * kv_len + j] : log_sp_pos);
        }
    }

    for (int i = 0; i < q_len; i++) {
        for (int j = 0; j < kv_len; j++) {
            int bkt = rel_bkt[i * kv_len + j];
            for (int h = 0; h < n_heads; h++) {
                out[i * n_heads * kv_len + h * kv_len + j] = pos_bias_w[bkt * n_heads + h];
            }
        }
    }

    free(rel_pos);
    free(rel_bkt);
}

void multi_head_attention(float *out, float *memory, float *state, float *bias, float *wq, float *wk, float *wv,
                          float *wo, int q_len, int kv_len, AttentionState *attn_state, ChronosBoltConfig *config) {
    // Multi-head attention
    // memory: (kv_len, d_model) the attented-to states, same as `state` for self-attention
    // state: (q_len, d_model) the attending states
    // bias: (q_len, n_heads, kv_len) the additive mask or position bias
    // wq: (n_heads * d_kv, d_model)
    // wk: (n_heads * d_kv, d_model)
    // wv: (n_heads * d_kv, d_model)
    // wo: (d_model, n_heads * d_kv)
    // out: (q_len, d_model)

    int d_model = config->d_model;
    int d_kv = config->d_kv;
    int n_heads = config->n_heads;

    float *query = attn_state->query;
    float *key = attn_state->key;
    float *value = attn_state->value;
    float *score = attn_state->score;
    float *attn_out = attn_state->attn_out;

    memset(attn_out, 0, (size_t)q_len * n_heads * d_kv * sizeof(float));

    #pragma omp parallel for
    for (int t = 0; t < q_len; t++) {
        matmul(query + t * n_heads * d_kv, wq, state + t * d_model, n_heads * d_kv, d_model);
    }
    #pragma omp parallel for
    for (int t = 0; t < kv_len; t++) {
        matmul(key + t * n_heads * d_kv, wk, memory + t * d_model, n_heads * d_kv, d_model);
        matmul(value + t * n_heads * d_kv, wv, memory + t * d_model, n_heads * d_kv, d_model);
    }
    #pragma omp parallel for collapse(2)
    for (int t = 0; t < q_len; t++) {
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < kv_len; i++) {
                matmul(score + t * n_heads * kv_len + h * kv_len + i, query + t * n_heads * d_kv + h * d_kv,
                       key + i * n_heads * d_kv + h * d_kv, 1, d_kv);
            }
        }
    }
    if (bias != NULL) {
        add(score, score, bias, q_len * n_heads * kv_len);
    }
    #pragma omp parallel for
    for (int i = 0; i < q_len * n_heads; i++) {
        softmax(score + i * kv_len, kv_len);
    }
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < q_len; i++) {
        for (int j = 0; j < n_heads; j++) {
            for (int k = 0; k < kv_len; k++) {
                for (int l = 0; l < d_kv; l++) {
                    attn_out[i * n_heads * d_kv + j * d_kv + l] +=
                        score[i * n_heads * kv_len + j * kv_len + k] * value[k * n_heads * d_kv + j * d_kv + l];
                }
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < q_len; i++) {
        matmul(out + i * d_model, wo, attn_out + i * n_heads * d_kv, d_model, n_heads * d_kv);
    }
}

void instance_norm(float *x, int size, float *loc, float *scale) {
    // Scaling

    float mean = 0.0f;
    float std = 0.0f;

    #pragma omp parallel for reduction(+ : mean)
    for (int i = 0; i < size; i++) {
        mean += x[i];
    }
    mean /= size;

    #pragma omp parallel for reduction(+ : std)
    for (int i = 0; i < size; i++) {
        std += (x[i] - mean) * (x[i] - mean);
    }
    std = sqrt(std / size);

    if (std == 0.0f) {
        std = 1e-5;
    }

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] - mean) / std;
    }

    *loc = mean;
    *scale = std;
}

void instance_denorm(float *x, int size, float loc, float scale) {
    // Un-scaling

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] = x[i] * scale + loc;
    }
}

void relu(float *x, int size) {
    // Element-wise ReLU

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(x[i], 0.0f);
    }
}

void patch_embedding(float *inp, float *out, float *hid_w, float *hid_b, float *out_w, float *out_b, float *res_w,
                     float *res_b, int n, int in_dim, int hid_dim, int out_dim) {
    float *buf1 = (float *)malloc(hid_dim * sizeof(float));
    float *buf2 = (float *)malloc(out_dim * sizeof(float));

    for (int i = 0; i < n; i++) {
        float *curr_inp = inp + i * in_dim;
        float *curr_out = out + i * out_dim;

        matmul_with_bias(buf1, hid_w, hid_b, curr_inp, hid_dim, in_dim);
        relu(buf1, hid_dim);
        matmul_with_bias(buf2, out_w, out_b, buf1, out_dim, hid_dim);
        matmul_with_bias(curr_out, res_w, res_b, curr_inp, out_dim, in_dim);
        add(curr_out, curr_out, buf2, out_dim);
    }

    free(buf1);
    free(buf2);
}

// Main prediction function

void predict(ChronosBolt *chronos_bolt, TimeSeries *ts, float *forecast) {
    const int d_model = chronos_bolt->config.d_model;
    const int d_ff = chronos_bolt->config.d_ff;
    const float layer_norm_eps = chronos_bolt->config.layer_norm_eps;
    const int n_heads = chronos_bolt->config.n_heads;
    const int d_kv = chronos_bolt->config.d_kv;

    float *x = (float *)malloc(ts->size * sizeof(float));
    if (x == NULL) {
        fprintf(stderr, "Error: could not allocate memory for x\n");
        exit(EXIT_FAILURE);
    }
    memcpy(x, ts->target, ts->size * sizeof(float));
    // Instance normalization
    float loc, scale;
    instance_norm(x, ts->size, &loc, &scale);

    // Check if padding is needed so that inputs are multiples of patch_size
    int patch_size = chronos_bolt->config.input_patch_size;
    int padding_size = 0;
    int rem = ts->size % patch_size;
    if (rem != 0) {
        padding_size = patch_size - rem;
    }
    int inp_mask_size = 2 * (padding_size + ts->size);
    int num_patches = (ts->size + padding_size) / patch_size;
    // xm holds the patched target and mask (after any necessary padding)
    float *xm = (float *)malloc(inp_mask_size * sizeof(float));
    if (xm == NULL) {
        fprintf(stderr, "Error: could not allocate memory for xm\n");
        exit(EXIT_FAILURE);
    }
    int patch_idx = 0;
    int target_idx = 0;

    if (padding_size > 0) {
        int i = 0;
        // Fill the padding and corresponding mask elements with zeros
        for (i = 0; i < padding_size; i++) {
            xm[i] = 0.0f;
            xm[i + patch_size] = 0.0f;
        }
        // Fill the remaining part of the first patch with actual values
        // and the mask positions with ones
        for (; target_idx < rem; target_idx++) {
            xm[i + target_idx] = x[target_idx];
            xm[i + target_idx + patch_size] = 1.0f;
        }
        patch_idx = 1;
    }
    // Fill all the remaining patches with data and the masks with ones
    // NOTE: Assumes that there are no NaNs aka missing values
    for (; patch_idx < num_patches; patch_idx++) {
        for (int j = 0; j < patch_size; j++, target_idx++) {
            int data_idx = 2 * patch_idx * patch_size + j;
            xm[data_idx] = x[target_idx];
            xm[data_idx + patch_size] = 1.0f;
        }
    }

    // Allocate memory for buffers
    RunState *rs = &chronos_bolt->run_state;
    malloc_run_state(num_patches, rs, &chronos_bolt->config);

    // Construct patch embeddings for inputs
    patch_embedding(xm, rs->embeds, chronos_bolt->ws.inp_patch_emb_hid_w, chronos_bolt->ws.inp_patch_emb_hid_b,
                    chronos_bolt->ws.inp_patch_emb_out_w, chronos_bolt->ws.inp_patch_emb_out_b,
                    chronos_bolt->ws.inp_patch_emb_res_w, chronos_bolt->ws.inp_patch_emb_res_b, num_patches,
                    chronos_bolt->config.input_patch_size * 2, chronos_bolt->config.d_ff,
                    chronos_bolt->config.d_model);

    // Append [REG] token to patch embeddings
    float *reg_token_dst = rs->embeds + num_patches * d_model;
    float *reg_token_src = chronos_bolt->ws.spl_tokens + d_model;
    memcpy(reg_token_dst, reg_token_src, d_model * sizeof(float));

    // Encoder

    t5_relative_pos_bias(rs->enc_pos_bias, chronos_bolt->ws.encoder_rel_pos_bias, num_patches + 1, num_patches + 1,
                         true, &chronos_bolt->config);
    // Allocate buffers for MHA
    AttentionState *enc_attn_state = malloc(sizeof(AttentionState));
    malloc_attn_state(num_patches + 1, num_patches + 1, enc_attn_state, &chronos_bolt->config);
    // Loop over encoder layers
    for (int l = 0; l < chronos_bolt->config.n_encoder_layers; l++) {
        // RMS Norm
        batched_rmsnorm(rs->buf_d, rs->embeds, chronos_bolt->ws.encoder_self_attn_ln + l * d_model, d_model,
                        num_patches + 1, layer_norm_eps);
        // MHA
        multi_head_attention(rs->buf_d, rs->buf_d, rs->buf_d, rs->enc_pos_bias,
                             chronos_bolt->ws.encoder_self_attn_q + l * n_heads * d_kv * d_model,
                             chronos_bolt->ws.encoder_self_attn_k + l * n_heads * d_kv * d_model,
                             chronos_bolt->ws.encoder_self_attn_v + l * n_heads * d_kv * d_model,
                             chronos_bolt->ws.encoder_self_attn_o + l * d_model * n_heads * d_kv, num_patches + 1,
                             num_patches + 1, enc_attn_state, &chronos_bolt->config);
        // Residual Connection
        add(rs->embeds, rs->embeds, rs->buf_d, (num_patches + 1) * d_model);
        // FF RMS Norm
        batched_rmsnorm(rs->buf_d, rs->embeds, chronos_bolt->ws.encoder_ff_ln + l * d_model, d_model, num_patches + 1,
                        layer_norm_eps);
        // WI
        batched_matmul(rs->buf_h, chronos_bolt->ws.encoder_ff_wi + l * d_ff * d_model, rs->buf_d, d_ff, d_model,
                       num_patches + 1);
        // ReLU
        relu(rs->buf_h, (num_patches + 1) * d_ff);
        // WO
        batched_matmul(rs->buf_d, chronos_bolt->ws.encoder_ff_wo + l * d_model * d_ff, rs->buf_h, d_model, d_ff,
                       num_patches + 1);
        // Residual Connection
        add(rs->embeds, rs->embeds, rs->buf_d, (num_patches + 1) * d_model);
    }
    batched_rmsnorm(rs->embeds, rs->embeds, chronos_bolt->ws.encoder_final_ln, d_model, num_patches + 1,
                    layer_norm_eps);

    free_attn_state(enc_attn_state);
    free(enc_attn_state);

    // Decoder
    memcpy(rs->dec_embeds, chronos_bolt->ws.spl_tokens, d_model * sizeof(float));

    t5_relative_pos_bias(rs->dec_pos_bias, chronos_bolt->ws.decoder_rel_pos_bias, 1, 1, false, &chronos_bolt->config);
    // Allocate buffers for MHA
    AttentionState *dec_self_attn_state = malloc(sizeof(AttentionState));
    malloc_attn_state(1, 1, dec_self_attn_state, &chronos_bolt->config);
    AttentionState *dec_cross_attn_state = malloc(sizeof(AttentionState));
    malloc_attn_state(1, num_patches + 1, dec_cross_attn_state, &chronos_bolt->config);
    // Loop over decoder layers
    for (int l = 0; l < chronos_bolt->config.n_decoder_layers; l++) {
        // Self-Attention
        // RMS Norm
        batched_rmsnorm(rs->dec_buf_d, rs->dec_embeds, chronos_bolt->ws.decoder_self_attn_ln + l * d_model, d_model, 1,
                        layer_norm_eps);
        // MHA
        multi_head_attention(rs->dec_buf_d, rs->dec_buf_d, rs->dec_buf_d, rs->dec_pos_bias,
                             chronos_bolt->ws.decoder_self_attn_q + +l * n_heads * d_kv * d_model,
                             chronos_bolt->ws.decoder_self_attn_k + +l * n_heads * d_kv * d_model,
                             chronos_bolt->ws.decoder_self_attn_v + l * n_heads * d_kv * d_model,
                             chronos_bolt->ws.decoder_self_attn_o + l * d_model * n_heads * d_kv, 1, 1,
                             dec_self_attn_state, &chronos_bolt->config);
        // Residual Connection
        add(rs->dec_embeds, rs->dec_embeds, rs->dec_buf_d, d_model);

        // Cross-Attention
        // RMS Norm
        batched_rmsnorm(rs->dec_buf_d, rs->dec_embeds, chronos_bolt->ws.decoder_cross_attn_ln + l * d_model, d_model,
                        1, layer_norm_eps);
        // MHA
        multi_head_attention(rs->dec_buf_d, rs->embeds, rs->dec_buf_d, NULL,
                             chronos_bolt->ws.decoder_cross_attn_q + l * n_heads * d_kv * d_model,
                             chronos_bolt->ws.decoder_cross_attn_k + l * n_heads * d_kv * d_model,
                             chronos_bolt->ws.decoder_cross_attn_v + l * n_heads * d_kv * d_model,
                             chronos_bolt->ws.decoder_cross_attn_o + l * d_model * n_heads * d_kv, 1, num_patches + 1,
                             dec_cross_attn_state, &chronos_bolt->config);
        // Residual Connection
        add(rs->dec_embeds, rs->dec_embeds, rs->dec_buf_d, d_model);
        // FF RMS Norm
        batched_rmsnorm(rs->dec_buf_d, rs->dec_embeds, chronos_bolt->ws.decoder_ff_ln + l * d_model, d_model, 1,
                        layer_norm_eps);
        // WI
        matmul(rs->dec_buf_h, chronos_bolt->ws.decoder_ff_wi + l * d_ff * d_model, rs->dec_buf_d, d_ff, d_model);
        // ReLU
        relu(rs->dec_buf_h, d_ff);
        // WO
        matmul(rs->dec_buf_d, chronos_bolt->ws.decoder_ff_wo + l * d_model * d_ff, rs->dec_buf_h, d_model, d_ff);
        // Residual Connection
        add(rs->dec_embeds, rs->dec_embeds, rs->dec_buf_d, d_model);
    }

    batched_rmsnorm(rs->dec_embeds, rs->dec_embeds, chronos_bolt->ws.decoder_final_ln, d_model, 1, layer_norm_eps);

    free_attn_state(dec_self_attn_state);
    free_attn_state(dec_cross_attn_state);
    free(dec_self_attn_state);
    free(dec_cross_attn_state);

    patch_embedding(rs->dec_embeds, forecast, chronos_bolt->ws.out_patch_emb_hid_w,
                    chronos_bolt->ws.out_patch_emb_hid_b, chronos_bolt->ws.out_patch_emb_out_w,
                    chronos_bolt->ws.out_patch_emb_out_b, chronos_bolt->ws.out_patch_emb_res_w,
                    chronos_bolt->ws.out_patch_emb_res_b, 1, chronos_bolt->config.d_model, chronos_bolt->config.d_ff,
                    NUM_QUANTILES * chronos_bolt->config.prediction_length);
    instance_denorm(forecast, NUM_QUANTILES * chronos_bolt->config.prediction_length, loc, scale);

    free(x);
    free(xm);
    free_run_state(rs);
}

void write_forecast(FILE *fp, char *item_id, float *forecast, int prediction_length) {
    // Write forecast in CSV format
    for (int i = 0; i < prediction_length; i++) {
        fprintf(fp, "%s", item_id);
        for (int j = 0; j < NUM_QUANTILES; j++) {
            fprintf(fp, ",%.4f", forecast[j * prediction_length + i]);
        }
        fprintf(fp, "\n");
    }
}

void print_array(float *x, int size) {
    // Print utility for debugging
    if (size <= 6) {
        for (int i = 0; i < size; i++) {
            printf("%.4f ", x[i]);
        }
    } else {
        for (int i = 0; i < 3; i++) {
            printf("%.4f ", x[i]);
        }
        printf("... ");
        for (int i = size - 3; i < size; i++) {
            printf("%.4f ", x[i]);
        }
    }
    printf("\n");
}

void error_usage() {
    fprintf(stderr, "Usage:   chronos_bolt <checkpoint> <past_data> <forecast>\n");
    fprintf(stderr, "Example: ./chronos_bolt autogluon-chronos-bolt-small.bin data.csv forecast.csv\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *ckpt_path = NULL;
    char *past_data_path = NULL;
    char *forecast_path = NULL;

    if (argc != 4) {
        error_usage();
    } else {
        ckpt_path = argv[1];
        past_data_path = argv[2];
        forecast_path = argv[3];
    }

    ChronosBolt chronos_bolt;
    load_checkpoint(ckpt_path, &chronos_bolt);

    TimeSeries *ts_array = NULL;
    int num_series = read_csv(past_data_path, &ts_array);
    if (num_series > 0) {
        FILE *fp = fopen(forecast_path, "w");
        if (fp == NULL) {
            fprintf(stderr, "Error: could not open file %s\n", forecast_path);
            exit(EXIT_FAILURE);
        }

        fprintf(fp, "item_id");
        for (int i = 0; i < NUM_QUANTILES; i++) {
            fprintf(fp, ",q0.%d", i + 1);
        }
        fprintf(fp, "\n");

        float *forecast = (float *)malloc(NUM_QUANTILES * chronos_bolt.config.prediction_length * sizeof(float));
        for (int i = 0; i < num_series; i++) {
            printf("Forecasting time series %d of %d\n", i + 1, num_series);
            TimeSeries current_series = ts_array[i];
            predict(&chronos_bolt, &current_series, forecast);
            write_forecast(fp, current_series.item_id, forecast, chronos_bolt.config.prediction_length);
        }
        free_time_series(ts_array, num_series);
        free(forecast);
        fclose(fp);
    }

    free_weights(&chronos_bolt.ws);

    return 0;
}
