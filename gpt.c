/*
 * gpt.c
 *
 * A naive, from‐scratch C implementation of a GPT‑2–style transformer.
 *
 * This code loads a safetensors file (pretrained weights) and a tokenized
 * data file, then performs a forward pass (and, if training, a backward pass)
 * using straightforward, unoptimized implementations of elementary operations.
 *
 * All required files ("enc", "data", and "model.safetensors") must be in
 * the same directory as this source file.
 *
 * Compile with:
 *    gcc -std=c11 gpt.c -o gpt -lm
 *
 * DISCLAIMER: This code is pretty stupid for educational and worst–case performance evaluation purposes.
 */

 #include <assert.h>
 #include <limits.h>
 #include <math.h>
 #include <stdbool.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <sys/stat.h>
 #include <time.h>
 #include <unistd.h>
 #include <stdint.h>   // For uint32_t, uint16_t, uint64_t
 
// I have no idea why my compiler wasn't reading stdlib.h, so here's the implementation lol
 void *my_aligned_alloc(size_t alignment, size_t size) {
     // Overallocate: add alignment-1 plus extra space to store original pointer.
     void *ptr = malloc(size + alignment - 1 + sizeof(void*));
     if (!ptr) return NULL;
     uintptr_t raw = (uintptr_t)ptr + sizeof(void*);
     uintptr_t aligned = (raw + alignment - 1) & ~(alignment - 1);
     ((void**)aligned)[-1] = ptr; // Store the original pointer
     return (void*)aligned;
 }
 
 void my_aligned_free(void *aligned_ptr) {
     if (aligned_ptr) {
         free(((void**)aligned_ptr)[-1]);
     }
 }
 
//random things i had to add cause this won't compile, remove at your own risk lmao
 #ifndef M_SQRT1_2
 #define M_SQRT1_2 0.70710678118654752440
 #endif
 
 /* === Macros (Model and Data Dimensions) === */
 #define VALIDATE_PERFORMANCE 1
 
 #define ENC_FILE_SIZE 722883
 #define MAX_DATA_SIZE 1000000
 #define SAFETENSOR_FILE_SIZE 548105171
 #define SAFETENSOR_JSON_SIZE 14283
 
 #define VOCAB_SIZE 50257
 #define SEQUENCE_LENGTH 1024
 #define MODEL_DIM 768
 #define HEAD_DIM 64
 #define NUM_HEADS 12
 #define NUM_LAYERS 12
 #define INV_SQRT_HEAD_DIM 0.125f  // 1/sqrt(64)
 
 /* === Structures for Token Decoder === */
 struct DecoderItem {
     uint32_t offset;
     uint32_t size;
 };
 
 struct TokenDecoder {
     struct DecoderItem items[VOCAB_SIZE];
     char raw[ENC_FILE_SIZE - VOCAB_SIZE * sizeof(struct DecoderItem)];
 };
 
 /* === Model Parameters Structure === */
 struct ModelParameters {
     struct { float *weight; } tokenEmbedding;    
     struct { float *weight; } positionEmbedding;  
     struct {
         struct { float *bias; float *weight; } norm1;  
         struct {
             struct { float *bias; float *weight; } attentionCombined; 
             struct { float *bias; float *weight; } attentionProjection; 
         } attention;
         struct { float *bias; float *weight; } norm2; 
         struct {
             struct { float *bias; float *weight; } mlpFC;   
             struct { float *bias; float *weight; } mlpProj;   
         } mlp;
     } layers[NUM_LAYERS];
     struct { float *bias; float *weight; } finalNorm; // ln_f
 };
 
 /* === Gradients Structure (mirrors ModelParameters) === */
 struct Gradients {
     struct { float weight[VOCAB_SIZE][MODEL_DIM]; } tokenEmbedding;
     struct { float weight[SEQUENCE_LENGTH][MODEL_DIM]; } positionEmbedding;
     struct {
         struct {
             float weight[MODEL_DIM];
             float bias[MODEL_DIM];
         } norm1;
         struct {
             struct {
                 float weight[MODEL_DIM][3 * MODEL_DIM];
                 float bias[3 * MODEL_DIM];
             } attentionCombined;
             struct {
                 float weight[MODEL_DIM][MODEL_DIM];
                 float bias[MODEL_DIM];
             } attentionProjection;
         } attention;
         struct {
             float weight[MODEL_DIM];
             float bias[MODEL_DIM];
         } norm2;
         struct {
             struct {
                 float weight[MODEL_DIM][4 * MODEL_DIM];
                 float bias[4 * MODEL_DIM];
             } mlpFC;
             struct {
                 float weight[4 * MODEL_DIM][MODEL_DIM];
                 float bias[MODEL_DIM];
             } mlpProj;
         } mlp;
     } layers[NUM_LAYERS];
     struct { float weight[MODEL_DIM]; float bias[MODEL_DIM]; } finalNorm;
 };
 
 /* === Activations (Forward Pass) === */
 struct Activations {
     struct {
         float out[SEQUENCE_LENGTH][MODEL_DIM];
     } embedding;
     struct {
         struct {
             float rStd[SEQUENCE_LENGTH];
             float mean[SEQUENCE_LENGTH];
             float out[SEQUENCE_LENGTH][MODEL_DIM];
         } norm1;
         struct {
             struct {
                 float out[SEQUENCE_LENGTH][3 * MODEL_DIM];
             } attentionCombined;
             struct {
                 float out[NUM_HEADS][SEQUENCE_LENGTH][SEQUENCE_LENGTH];
             } softmax;
             struct {
                 float out[SEQUENCE_LENGTH][MODEL_DIM];
             } attentionOutput;
             struct {
                 float out[SEQUENCE_LENGTH][MODEL_DIM];
             } attentionProjection;
         } attention;
         struct {
             float out[SEQUENCE_LENGTH][MODEL_DIM];
         } residual1;
         struct {
             float rStd[SEQUENCE_LENGTH];
             float mean[SEQUENCE_LENGTH];
             float out[SEQUENCE_LENGTH][MODEL_DIM];
         } norm2;
         struct {
             struct {
                 float out[SEQUENCE_LENGTH][4 * MODEL_DIM];
             } mlpFC;
             struct {
                 float out[SEQUENCE_LENGTH][4 * MODEL_DIM];
             } gelu;
             struct {
                 float out[SEQUENCE_LENGTH][MODEL_DIM];
             } mlpProjection;
         } mlp;
         struct {
             float out[SEQUENCE_LENGTH][MODEL_DIM];
         } residual2;
     } layers[NUM_LAYERS];
     struct {
         float rStd[SEQUENCE_LENGTH];
         float mean[SEQUENCE_LENGTH];
         float out[SEQUENCE_LENGTH][MODEL_DIM];
     } finalNorm;
     struct {
         float out[SEQUENCE_LENGTH][VOCAB_SIZE];
     } unembedding;
 };
 
 /* === Backward Activations Structure === */
 struct BackwardActivations {
     struct {
         float out[SEQUENCE_LENGTH][MODEL_DIM];
     } embedding;
     struct {
         float out[SEQUENCE_LENGTH][MODEL_DIM];
     } norm1;
     struct {
         struct {
             float out[SEQUENCE_LENGTH][3 * MODEL_DIM];
         } attentionCombined;
         struct {
             float out[SEQUENCE_LENGTH];
         } softmax;
         struct {
             float out[SEQUENCE_LENGTH][MODEL_DIM];
         } attentionOutput;
     } attention;
     struct {
         float inResidual[SEQUENCE_LENGTH][MODEL_DIM];
         float out[SEQUENCE_LENGTH][MODEL_DIM];
     } residual1;
     struct {
         float out[SEQUENCE_LENGTH][MODEL_DIM];
     } norm2;
     struct {
         struct {
             float out[SEQUENCE_LENGTH][4 * MODEL_DIM];
         } mlpFC;
         struct {
             float out[SEQUENCE_LENGTH][4 * MODEL_DIM];
         } gelu;
     } mlp;
     struct {
         float inResidual[SEQUENCE_LENGTH][MODEL_DIM];
         float out[SEQUENCE_LENGTH][MODEL_DIM];
     } residual2;
     struct {
         float out[SEQUENCE_LENGTH][MODEL_DIM];
     } finalNorm;
     struct {
         float out[SEQUENCE_LENGTH][VOCAB_SIZE];
     } unembedding;
 };
 
 /* === Helper Structures for Elementary Operations === */
 struct ElementwiseAdd {
     const float* input1;
     const float* input2;
     float* output;
     size_t count;
 };
 
 struct FullyConnectedLayer {
     const float* weight; // [in_features x out_features] row-major
     const float* bias;   // [out_features]
     const float* input;  // [sample_count x in_features]
     float* output;       // [sample_count x out_features]
     size_t in_features;
     size_t out_features;
     size_t sample_count;
 };
 
 struct FullyConnectedLayerBackward {
     const float* weight;
     const float* input;
     const float* dL_doutput;
     float* dL_dweight;
     float* dL_dbias;
     float* dL_dinput;
     size_t in_features;
     size_t out_features;
     size_t sample_count;
 };
 
 struct LayerNorm {
     const float* gamma; // scale [in_features]
     const float* beta;  // bias [in_features]
     const float* input; // [sample_count x in_features]
     float* rStd;        // [sample_count]
     float* mean;        // [sample_count]
     float* output;      // [sample_count x in_features]
     size_t in_features;
     size_t sample_count;
 };
 
 struct LayerNormBackward {
     const float* gamma;
     const float* input;
     const float* rStd;
     const float* mean;
     const float* dL_doutput;
     float* dL_dgamma;   // [in_features]
     float* dL_dbias;    // [in_features]
     float* dL_dinput;   // [sample_count x in_features]
     size_t in_features;
     size_t sample_count;
 };
 
 /* === Validation Timing Structure === */
 struct ValidationTimes {
     double t_start;
     double t_last;
     double embedding;
     double norm1;
     struct {
         double attentionCombined;
         double attentionOutput;
         double attentionProjection;
     } attention;
     double residual1;
     double norm2;
     struct {
         double mlpFC;
         double gelu;
         double mlpProjection;
     } mlp;
     double residual2;
     double finalNorm;
     double unembedding;
     double total;
 };
 
 /* === Utility Functions === */
 static double get_current_time(void) {
     struct timespec t;
     clock_gettime(CLOCK_REALTIME, &t);
     return t.tv_sec + t.tv_nsec * 1e-9;
 }
 
 static void update_validation_time(double* target, double* last_time) {
     if (VALIDATE_PERFORMANCE) {
         double t = get_current_time();
         *target += t - *last_time;
         *last_time = t;
     }
 }
 
 static void validate_sum(float* array, size_t count, double expected_sum) {
     if (VALIDATE_PERFORMANCE) {
         double sum = 0.0;
         for (size_t i = 0; i < count; i++) {
             sum += array[i];
         }
         if (expected_sum != sum) {
             fprintf(stderr, "Expected sum: %.24f, got %.24f\n", expected_sum, sum);
             abort();
         }
     }
 }
 
 static void get_offset_and_size(const char* json_raw, const char* tensor_name, size_t* out_offset, size_t* out_size) {
     char temp[32];
     char* start = strstr(json_raw, tensor_name);
     start = strstr(start, "data_offsets") + 15;
     char* end = strstr(start, ",");
     memcpy(temp, start, end - start);
     temp[end - start] = '\0';
     size_t offset = (size_t)atoi(start);
     start = end + 1;
     end = strstr(start, "]");
     memcpy(temp, start, end - start);
     temp[end - start] = '\0';
     size_t offset_end = (size_t)atoi(temp);
     size_t size = offset_end - offset;
     *out_offset = offset;
     *out_size = size;
 }
 
 /* === Elementary Operation Functions === */
 static void elementwise_add(const struct ElementwiseAdd* add) {
     for (size_t i = 0; i < add->count; i++) {
         add->output[i] = add->input1[i] + add->input2[i];
     }
 }
 
 static void fully_connected(const struct FullyConnectedLayer* fc) {
     for (size_t sample = 0; sample < fc->sample_count; sample++) {
         float* out = (float*)fc->output + sample * fc->out_features;
         memcpy(out, fc->bias, fc->out_features * sizeof(float));
         for (size_t in = 0; in < fc->in_features; in++) {
             float input_val = fc->input[sample * fc->in_features + in];
             for (size_t j = 0; j < fc->out_features; j++) {
                 out[j] += input_val * fc->weight[in * fc->out_features + j];
             }
         }
     }
 }
 
 static void fully_connected_backward(const struct FullyConnectedLayerBackward* fc) {
     for (size_t sample = 0; sample < fc->sample_count; sample++) {
         const float* dL_dout = fc->dL_doutput + sample * fc->out_features;
         for (size_t j = 0; j < fc->out_features; j++) {
             fc->dL_dbias[j] += dL_dout[j];
         }
         const float* input_ptr = fc->input + sample * fc->in_features;
         float* dL_din = fc->dL_dinput + sample * fc->in_features;
         for (size_t i = 0; i < fc->in_features; i++) {
             for (size_t j = 0; j < fc->out_features; j++) {
                 fc->dL_dweight[i * fc->out_features + j] += input_ptr[i] * dL_dout[j];
                 dL_din[i] += fc->weight[i * fc->out_features + j] * dL_dout[j];
             }
         }
     }
 }
 
 static void layer_norm(const struct LayerNorm* ln) {
     for (size_t sample = 0; sample < ln->sample_count; sample++) {
         const float* input_ptr = ln->input + sample * ln->in_features;
         float sum = 0.0f;
         for (size_t i = 0; i < ln->in_features; i++) {
             sum += input_ptr[i];
         }
         float mean = sum / ln->in_features;
         ln->mean[sample] = mean;
         float sum_sq_diff = 0.0f;
         for (size_t i = 0; i < ln->in_features; i++) {
             float diff = input_ptr[i] - mean;
             sum_sq_diff += diff * diff;
         }
         float variance = sum_sq_diff / ln->in_features;
         float rStd = 1.0f / sqrtf(variance + 1e-5f);
         ln->rStd[sample] = rStd;
         float* out_ptr = ln->output + sample * ln->in_features;
         for (size_t i = 0; i < ln->in_features; i++) {
             float normalized = (input_ptr[i] - mean) * rStd;
             out_ptr[i] = normalized * ln->gamma[i] + ln->beta[i];
         }
     }
 }
 
 static void layer_norm_backward(const struct LayerNormBackward* ln) {
     /* Naively implemented backward pass for layer norm.
        NOTE: This is a simplified version. */
     for (size_t sample = 0; sample < ln->sample_count; sample++) {
         float rStd = ln->rStd[sample];
         float mean = ln->mean[sample];
         const float* dL_dout_ptr = ln->dL_doutput + sample * ln->in_features;
         const float* input_ptr = ln->input + sample * ln->in_features;
         float* dL_din_ptr = ln->dL_dinput + sample * ln->in_features;
         for (size_t i = 0; i < ln->in_features; i++) {
             float normalized = (input_ptr[i] - mean) * rStd;
             dL_din_ptr[i] = dL_dout_ptr[i] * rStd;
         }
     }
 }
 
 /* === Validation Time Dump (for debugging) === */
 static void dump_validation_times(const struct ValidationTimes* vt) {
     printf("Validation Times:\n");
     printf("  Total: %.6f s\n", vt->total);
     printf("  Embedding: %.6f s\n", vt->embedding);
     printf("  Norm1: %.6f s\n", vt->norm1);
     printf("  Attention Combined: %.6f s\n", vt->attention.attentionCombined);
     printf("  Attention Output: %.6f s\n", vt->attention.attentionOutput);
     printf("  Attention Projection: %.6f s\n", vt->attention.attentionProjection);
     printf("  Residual1: %.6f s\n", vt->residual1);
     printf("  Norm2: %.6f s\n", vt->norm2);
     printf("  MLP FC: %.6f s\n", vt->mlp.mlpFC);
     printf("  GELU: %.6f s\n", vt->mlp.gelu);
     printf("  MLP Projection: %.6f s\n", vt->mlp.mlpProjection);
     printf("  Residual2: %.6f s\n", vt->residual2);
     printf("  Final Norm: %.6f s\n", vt->finalNorm);
     printf("  Unembedding: %.6f s\n", vt->unembedding);
 }
 
 /* === Core Transformer Process Function ===
  *
  * Performs a full forward pass (and, if training, a backward pass) on the input sequence.
  *
  * Parameters:
  *   modelParams        - Pointer to model parameters (loaded from safetensors)
  *   activations        - Buffer for forward activations
  *   inputTokens        - Array of input token IDs (uint16_t)
  *   inputSize          - Number of tokens in input sequence
  *   outputToken        - Pointer to a uint16_t where the generated token will be stored (in inference mode)
  *   isTraining         - True for training mode (forward + backward), false for inference
  *   gradients          - Buffer for gradients (used in training)
  *   backwardActivations- Buffer for backward activations (used in training)
  *   expectedTokens     - Array of expected target tokens (used in training)
  */
 static void process_transformer(struct ModelParameters* modelParams,
                                 struct Activations* activations,
                                 uint16_t* inputTokens, size_t inputSize,
                                 uint16_t* outputToken, bool isTraining,
                                 struct Gradients* gradients,
                                 struct BackwardActivations* backwardActivations,
                                 uint16_t* expectedTokens) {
     struct ValidationTimes vt = {0};
     vt.t_start = get_current_time();
     vt.t_last = vt.t_start;
 
     /* --- Embedding (Token + Position) --- */
     for (size_t i = 0; i < inputSize; i++) {
         float* tokenEmb = modelParams->tokenEmbedding.weight + inputTokens[i] * MODEL_DIM;
         float* posEmb = modelParams->positionEmbedding.weight + i * MODEL_DIM;
         float* outEmb = &activations->embedding.out[i][0];
         for (size_t d = 0; d < MODEL_DIM; d++) {
             outEmb[d] = tokenEmb[d] + posEmb[d];
         }
     }
     update_validation_time(&vt.embedding, &vt.t_last);
     validate_sum((float*)activations->embedding.out, inputSize * MODEL_DIM, -0x1.e86f2c2adep+4);
 
     /* --- Transformer Layers --- */
     for (int layer = 0; layer < NUM_LAYERS; layer++) {
         /* Layer Norm 1 */
         const float* ln1Input = (layer == 0) ? (float*)activations->embedding.out
                                              : (float*)activations->layers[layer - 1].residual2.out;
         struct LayerNorm norm1 = {
             .gamma = modelParams->layers[layer].norm1.weight,
             .beta = modelParams->layers[layer].norm1.bias,
             .input = ln1Input,
             .rStd = activations->layers[layer].norm1.rStd,
             .mean = activations->layers[layer].norm1.mean,
             .output = (float*)activations->layers[layer].norm1.out,
             .in_features = MODEL_DIM,
             .sample_count = inputSize,
         };
         layer_norm(&norm1);
         update_validation_time(&vt.norm1, &vt.t_last);
         if (layer == 0)
             validate_sum((float*)activations->layers[layer].norm1.out, inputSize * MODEL_DIM, -0x1.4e34ee18da56ap+8);
 
         /* Attention: Combined FC for Q, K, V */
         struct FullyConnectedLayer fcAttnCombined = {
             .weight = modelParams->layers[layer].attention.attentionCombined.weight,
             .bias = modelParams->layers[layer].attention.attentionCombined.bias,
             .input = (float*)activations->layers[layer].norm1.out,
             .output = (float*)activations->layers[layer].attention.attentionCombined.out,
             .in_features = MODEL_DIM,
             .out_features = 3 * MODEL_DIM,
             .sample_count = inputSize,
         };
         fully_connected(&fcAttnCombined);
         update_validation_time(&vt.attention.attentionCombined, &vt.t_last);
         if (layer == 0)
             validate_sum((float*)activations->layers[layer].attention.attentionCombined.out, inputSize * 3 * MODEL_DIM, -0x1.9f967d2b7f151p+11);
 
         /* Zero out attention output accumulator */
         memset(activations->layers[layer].attention.attentionOutput.out, 0, sizeof(activations->layers[layer].attention.attentionOutput.out));
 
         /* Self-Attention per head (naively implemented) */
         for (size_t head = 0; head < NUM_HEADS; head++) {
             for (size_t q = 0; q < inputSize; q++) {
                 float* softmaxOut = activations->layers[layer].attention.softmax.out[head][q];
                 float maxScore = -INFINITY;
                 for (size_t k = 0; k <= q; k++) {
                     float* query = (float*)activations->layers[layer].attention.attentionCombined.out + q * 3 * MODEL_DIM + head * HEAD_DIM;
                     float* key = (float*)activations->layers[layer].attention.attentionCombined.out + k * 3 * MODEL_DIM + MODEL_DIM + head * HEAD_DIM;
                     float dot = 0.0f;
                     for (size_t d = 0; d < HEAD_DIM; d++) {
                         dot += query[d] * key[d];
                     }
                     dot *= INV_SQRT_HEAD_DIM;
                     softmaxOut[k] = dot;
                     if (dot > maxScore)
                         maxScore = dot;
                 }
                 float sumExp = 0.0f;
                 for (size_t k = 0; k <= q; k++) {
                     float expVal = expf(softmaxOut[k] - maxScore);
                     sumExp += expVal;
                     softmaxOut[k] = expVal;
                 }
                 float invSumExp = 1.0f / sumExp;
                 for (size_t k = 0; k <= q; k++) {
                     softmaxOut[k] *= invSumExp;
                 }
                 /* Compute weighted sum of V vectors */
                 for (size_t v = 0; v <= q; v++) {
                     float* value = (float*)activations->layers[layer].attention.attentionCombined.out + v * 3 * MODEL_DIM + 2 * MODEL_DIM + head * HEAD_DIM;
                     float* outSlice = (float*)activations->layers[layer].attention.attentionOutput.out + q * MODEL_DIM + head * HEAD_DIM;
                     float weightVal = softmaxOut[v];
                     for (size_t d = 0; d < HEAD_DIM; d++) {
                         outSlice[d] += weightVal * value[d];
                     }
                 }
             }
         }
         update_validation_time(&vt.attention.attentionOutput, &vt.t_last);
         if (layer == 0)
             validate_sum((float*)activations->layers[layer].attention.attentionOutput.out, inputSize * MODEL_DIM, 0x1.c64a4db1bfcdep+8);
 
         /* Attention Projection */
         struct FullyConnectedLayer fcAttnProj = {
             .weight = modelParams->layers[layer].attention.attentionProjection.weight,
             .bias = modelParams->layers[layer].attention.attentionProjection.bias,
             .input = (float*)activations->layers[layer].attention.attentionOutput.out,
             .output = (float*)activations->layers[layer].attention.attentionProjection.out,
             .in_features = MODEL_DIM,
             .out_features = MODEL_DIM,
             .sample_count = inputSize,
         };
         fully_connected(&fcAttnProj);
         update_validation_time(&vt.attention.attentionProjection, &vt.t_last);
         if (layer == 0)
             validate_sum((float*)activations->layers[layer].attention.attentionProjection.out, inputSize * MODEL_DIM, 0x1.850b3ffab297bp+8);
 
         /* Residual Connection 1 */
         struct ElementwiseAdd addResidual1 = {
             .input1 = (layer == 0) ? (float*)activations->embedding.out : (float*)activations->layers[layer - 1].residual2.out,
             .input2 = (float*)activations->layers[layer].attention.attentionProjection.out,
             .output = (float*)activations->layers[layer].residual1.out,
             .count = inputSize * MODEL_DIM,
         };
         elementwise_add(&addResidual1);
         update_validation_time(&vt.residual1, &vt.t_last);
 
         /* Layer Norm 2 */
         struct LayerNorm norm2 = {
             .gamma = modelParams->layers[layer].norm2.weight,
             .beta = modelParams->layers[layer].norm2.bias,
             .input = (float*)activations->layers[layer].residual1.out,
             .rStd = activations->layers[layer].norm2.rStd,
             .mean = activations->layers[layer].norm2.mean,
             .output = (float*)activations->layers[layer].norm2.out,
             .in_features = MODEL_DIM,
             .sample_count = inputSize,
         };
         layer_norm(&norm2);
         update_validation_time(&vt.norm2, &vt.t_last);
         if (layer == 0)
             validate_sum((float*)activations->layers[layer].norm2.out, inputSize * MODEL_DIM, 0x1.188ffb5000f3dp+8);
 
         /* MLP: FC for intermediate projection */
         struct FullyConnectedLayer fcMlpFC = {
             .weight = modelParams->layers[layer].mlp.mlpFC.weight,
             .bias = modelParams->layers[layer].mlp.mlpFC.bias,
             .input = (float*)activations->layers[layer].norm2.out,
             .output = (float*)activations->layers[layer].mlp.mlpFC.out,
             .in_features = MODEL_DIM,
             .out_features = 4 * MODEL_DIM,
             .sample_count = inputSize,
         };
         fully_connected(&fcMlpFC);
         update_validation_time(&vt.mlp.mlpFC, &vt.t_last);
 
         /* GELU Activation */
         {
             const float* fcIn = (float*)activations->layers[layer].mlp.mlpFC.out;
             size_t totalElems = inputSize * 4 * MODEL_DIM;
             float* geluOut = (float*)activations->layers[layer].mlp.gelu.out;
             for (size_t i = 0; i < totalElems; i++) {
                 float phi = 0.5f * (1.0f + erff(fcIn[i] * M_SQRT1_2));
                 geluOut[i] = fcIn[i] * phi;
             }
         }
         update_validation_time(&vt.mlp.gelu, &vt.t_last);
 
         /* MLP: Projection FC */
         struct FullyConnectedLayer fcMlpProj = {
             .weight = modelParams->layers[layer].mlp.mlpProj.weight,
             .bias = modelParams->layers[layer].mlp.mlpProj.bias,
             .input = (float*)activations->layers[layer].mlp.gelu.out,
             .output = (float*)activations->layers[layer].mlp.mlpProjection.out,
             .in_features = 4 * MODEL_DIM,
             .out_features = MODEL_DIM,
             .sample_count = inputSize,
         };
         fully_connected(&fcMlpProj);
         update_validation_time(&vt.mlp.mlpProjection, &vt.t_last);
         if (layer == 0)
             validate_sum((float*)activations->layers[layer].mlp.mlpProjection.out, inputSize * MODEL_DIM, -0x1.012ce31d82fb8p+9);
 
         /* Residual Connection 2 */
         struct ElementwiseAdd addResidual2 = {
             .input1 = (float*)activations->layers[layer].residual1.out,
             .input2 = (float*)activations->layers[layer].mlp.mlpProjection.out,
             .output = (float*)activations->layers[layer].residual2.out,
             .count = inputSize * MODEL_DIM,
         };
         elementwise_add(&addResidual2);
         update_validation_time(&vt.residual2, &vt.t_last);
     }
 
     /* --- Final Layer Norm --- */
     struct LayerNorm finalNorm = {
         .gamma = modelParams->finalNorm.weight,
         .beta = modelParams->finalNorm.bias,
         .input = (float*)activations->layers[NUM_LAYERS - 1].residual2.out,
         .rStd = activations->finalNorm.rStd,
         .mean = activations->finalNorm.mean,
         .output = (float*)activations->finalNorm.out,
         .in_features = MODEL_DIM,
         .sample_count = inputSize,
     };
     layer_norm(&finalNorm);
     update_validation_time(&vt.finalNorm, &vt.t_last);
     validate_sum((float*)activations->finalNorm.out, inputSize * MODEL_DIM, 0x1.0437f5b8f47d8p+14);
 
     /* --- Unembedding --- */
     for (size_t i = (isTraining ? 0 : inputSize - 1); i < inputSize; i++) {
         float* logits = activations->unembedding.out[i];
         const float* embedWeights = modelParams->tokenEmbedding.weight;
         const float* embedWeightsEnd = embedWeights + VOCAB_SIZE * MODEL_DIM;
         const float* sampleVec = activations->finalNorm.out[i];
         float dot = 0.0f, maxLogit = -INFINITY;
         float* logitsPtr = logits;
         const float* weightPtr = embedWeights;
         const float* samplePtr = sampleVec;
         while (true) {
             dot += (*weightPtr) * (*samplePtr);
             weightPtr++;
             samplePtr++;
             if (samplePtr == sampleVec + MODEL_DIM) {
                 samplePtr = sampleVec;
                 *logitsPtr = dot;
                 if (dot > maxLogit) maxLogit = dot;
                 dot = 0.0f;
                 logitsPtr++;
                 if (weightPtr == embedWeightsEnd)
                     break;
             }
         }
         float sumExp = 0.0f;
         for (logitsPtr = logits; logitsPtr < logits + VOCAB_SIZE; logitsPtr++) {
             float expVal = expf(*logitsPtr - maxLogit);
             sumExp += expVal;
             *logitsPtr = expVal;
         }
         float invSumExp = 1.0f / sumExp;
         for (logitsPtr = logits; logitsPtr < logits + VOCAB_SIZE; logitsPtr++) {
             *logitsPtr *= invSumExp;
         }
     }
     update_validation_time(&vt.unembedding, &vt.t_last);
     validate_sum((float*)activations->unembedding.out, inputSize * VOCAB_SIZE, 0x1.0008be62ee50cp+6);
     update_validation_time(&vt.total, &vt.t_start);
     if (VALIDATE_PERFORMANCE)
         dump_validation_times(&vt);
 
     if (!isTraining) {
         const float* finalLogits = activations->unembedding.out[inputSize - 1];
         const float* bestPtr = finalLogits;
         for (const float* ptr = finalLogits; ptr < finalLogits + VOCAB_SIZE; ptr++) {
             if (*ptr > *bestPtr)
                 bestPtr = ptr;
         }
         *outputToken = (uint16_t)(bestPtr - finalLogits);
         return;
     }
 
     /* --- BACKWARD PASS (Naively implemented) ---
      * For brevity, only the unembedding and embedding backprop are shown.
      */
     memset(&vt, 0, sizeof(vt));
     vt.t_start = get_current_time();
     vt.t_last = vt.t_start;
 
     /* Unembedding backward: compute cross–entropy loss gradient */
     memcpy(backwardActivations->unembedding.out, activations->unembedding.out, inputSize * VOCAB_SIZE * sizeof(float));
     float totalLoss = 0.0f;
     for (size_t i = 0; i < inputSize; i++) {
         uint16_t correct = expectedTokens[i];
         float pCorrect = backwardActivations->unembedding.out[i][correct];
         backwardActivations->unembedding.out[i][correct] = pCorrect - 1.0f;
         totalLoss += -logf(pCorrect);
     }
     validate_sum(&totalLoss, 1, 0x1.08868ep+8);
     float invInputSize = 1.0f / (float)inputSize;
     memset(backwardActivations->finalNorm.out, 0, sizeof(backwardActivations->finalNorm.out));
     for (size_t i = 0; i < inputSize; i++) {
         float* dL_din_final = backwardActivations->finalNorm.out[i];
         const float* sampleVec = activations->finalNorm.out[i];
         const float* dL_dout = backwardActivations->unembedding.out[i];
         const float* embedWeights = modelParams->tokenEmbedding.weight;
         float* dW_token = gradients->tokenEmbedding.weight[inputTokens[i]];
         float* dW_pos = gradients->positionEmbedding.weight[i];
         while (true) {
             *dW_token += (*dL_dout) * (*sampleVec) * invInputSize;
             *dL_din_final += (*dL_dout) * (*embedWeights);
             embedWeights++;
             sampleVec++;
             dW_token++;
             if (sampleVec == activations->finalNorm.out[i] + MODEL_DIM) {
                 sampleVec = activations->finalNorm.out[i];
                 dL_din_final = backwardActivations->finalNorm.out[i];
                 dL_dout++;
                 if (dL_dout == backwardActivations->unembedding.out[i] + VOCAB_SIZE)
                     break;
             }
         }
         for (size_t j = 0; j < MODEL_DIM; j++) {
             dL_din_final[j] *= invInputSize;
         }
     }
     update_validation_time(&vt.unembedding, &vt.t_last);
     validate_sum((float*)gradients->tokenEmbedding.weight, VOCAB_SIZE * MODEL_DIM, 0x1.7f75a7f7bb39p-6);
     validate_sum((float*)gradients->positionEmbedding.weight, inputSize * MODEL_DIM, 0x1.e96f7cp-20);
     update_validation_time(&vt.embedding, &vt.t_last);
     update_validation_time(&vt.total, &vt.t_start);
     if (VALIDATE_PERFORMANCE)
         dump_validation_times(&vt);
 
     /* Finally, backpropagate into the embedding parameters */
     for (size_t i = 0; i < inputSize; i++) {
         const float* dL_dout = backwardActivations->embedding.out[i];
         float* dW_token = gradients->tokenEmbedding.weight[inputTokens[i]];
         float* dW_pos = gradients->positionEmbedding.weight[i];
         for (size_t j = 0; j < MODEL_DIM; j++) {
             dW_token[j] += dL_dout[j];
             dW_pos[j] += dL_dout[j];
         }
     }
     update_validation_time(&vt.embedding, &vt.t_last);
     validate_sum((float*)gradients->tokenEmbedding.weight, VOCAB_SIZE * MODEL_DIM, 0x1.7f75a7f7bb39p-6);
     validate_sum((float*)gradients->positionEmbedding.weight, inputSize * MODEL_DIM, 0x1.e96f7cp-20);
     update_validation_time(&vt.total, &vt.t_start);
     if (VALIDATE_PERFORMANCE)
         dump_validation_times(&vt);
 }
 
 /* === Main Function === */
 int main(void) {
     /* Calculate total memory required and allocate one big block */
     #define ALIGN(x) (((x) + 255) & ~(size_t)0xff)
     size_t totalMem = 0;
     size_t decoderOffset = totalMem; totalMem += ALIGN(sizeof(struct TokenDecoder));
     size_t tokenDataOffset = totalMem; totalMem += ALIGN(MAX_DATA_SIZE);
     size_t jsonOffset = totalMem; totalMem += ALIGN(SAFETENSOR_JSON_SIZE);
     size_t rawParamsOffset = totalMem; totalMem += ALIGN(SAFETENSOR_FILE_SIZE);
     size_t structuredParamsOffset = totalMem; totalMem += ALIGN(sizeof(struct ModelParameters));
     size_t activationsOffset = totalMem; totalMem += ALIGN(sizeof(struct Activations));
     size_t gradientsOffset = totalMem; totalMem += ALIGN(sizeof(struct Gradients));
     size_t backwardActivationsOffset = totalMem; totalMem += ALIGN(sizeof(struct BackwardActivations));
     #undef ALIGN
 
     fprintf(stderr, "Total memory required: %lu MiB\n", totalMem >> 20);
     char* memoryBlock = my_aligned_alloc(256, totalMem);
     assert(memoryBlock);
 
     struct TokenDecoder* decoder = (struct TokenDecoder*)(memoryBlock);
     uint16_t* tokenData = (uint16_t*)(memoryBlock + tokenDataOffset);
     char* jsonRaw = memoryBlock + jsonOffset;
     char* rawParams = memoryBlock + rawParamsOffset;
     struct ModelParameters* modelParams = (struct ModelParameters*)(memoryBlock + structuredParamsOffset);
     struct Activations* activations = (struct Activations*)(memoryBlock + activationsOffset);
     struct Gradients* gradients = (struct Gradients*)(memoryBlock + gradientsOffset);
     struct BackwardActivations* backwardActivations = (struct BackwardActivations*)(memoryBlock + backwardActivationsOffset);
 
     /* --- Load Token Decoder --- */
     {
         char encPath[PATH_MAX];
         snprintf(encPath, sizeof(encPath), "enc");
         FILE* fEnc = fopen(encPath, "rb");
         assert(fEnc);
         struct stat st;
         fstat(fileno(fEnc), &st);
         assert(st.st_size == ENC_FILE_SIZE);
         size_t readSize = VOCAB_SIZE * sizeof(struct DecoderItem);
         size_t bytesRead = fread(&decoder->items, 1, readSize, fEnc);
         assert(bytesRead == readSize);
         readSize = ENC_FILE_SIZE - readSize;
         bytesRead = fread(&decoder->raw, 1, readSize, fEnc);
         assert(bytesRead == readSize);
         fclose(fEnc);
         for (int i = 0; i < VOCAB_SIZE; i++) {
             assert(decoder->items[i].size != 0);
         }
     }
 
     /* --- Load Tokenized Data --- */
     size_t tokenCount;
     {
         char dataPath[PATH_MAX];
         snprintf(dataPath, sizeof(dataPath), "data");
         FILE* fData = fopen(dataPath, "rb");
         assert(fData);
         struct stat st;
         fstat(fileno(fData), &st);
         assert((size_t)st.st_size <= MAX_DATA_SIZE);
         assert((size_t)st.st_size % sizeof(uint16_t) == 0);
         tokenCount = st.st_size / sizeof(uint16_t);
         size_t bytesRead = fread(tokenData, 1, st.st_size, fData);
         assert(bytesRead == (size_t)st.st_size);
         fclose(fData);
     }
 
     /* --- Load Safetensor Parameters --- */
     {
         char paramsPath[PATH_MAX];
         snprintf(paramsPath, sizeof(paramsPath), "model.safetensors");
         FILE* fParams = fopen(paramsPath, "rb");
         assert(fParams);
         struct stat st;
         fstat(fileno(fParams), &st);
         assert((size_t)st.st_size == SAFETENSOR_FILE_SIZE);
         uint64_t jsonSize;
         size_t bytesRead = fread(&jsonSize, 1, sizeof(jsonSize), fParams);
         assert(bytesRead == sizeof(jsonSize));
         assert(jsonSize == SAFETENSOR_JSON_SIZE);
         bytesRead = fread(jsonRaw, 1, jsonSize, fParams);
         assert(bytesRead == jsonSize);
         long pos = ftell(fParams);
         assert(pos != -1);
         size_t rawSize = SAFETENSOR_FILE_SIZE - (size_t)pos;
         bytesRead = fread(rawParams, 1, rawSize, fParams);
         assert(bytesRead == rawSize);
 
         size_t offset, size;
         /* Load token embedding */
         get_offset_and_size(jsonRaw, "wte", &offset, &size);
         modelParams->tokenEmbedding.weight = (float*)(rawParams + offset);
         assert(size == VOCAB_SIZE * MODEL_DIM * sizeof(float));
 
         /* Load position embedding */
         get_offset_and_size(jsonRaw, "wpe", &offset, &size);
         modelParams->positionEmbedding.weight = (float*)(rawParams + offset);
         assert(size == SEQUENCE_LENGTH * MODEL_DIM * sizeof(float));
 
         for (int l = 0; l < NUM_LAYERS; l++) {
             char label[32];
             snprintf(label, sizeof(label), "h.%d.ln_1.bias", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].norm1.bias = (float*)(rawParams + offset);
             assert(size == MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.ln_1.weight", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].norm1.weight = (float*)(rawParams + offset);
             assert(size == MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.attn.c_attn.bias", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].attention.attentionCombined.bias = (float*)(rawParams + offset);
             assert(size == 3 * MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.attn.c_attn.weight", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].attention.attentionCombined.weight = (float*)(rawParams + offset);
             assert(size == MODEL_DIM * 3 * MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.attn.c_proj.bias", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].attention.attentionProjection.bias = (float*)(rawParams + offset);
             assert(size == MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.attn.c_proj.weight", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].attention.attentionProjection.weight = (float*)(rawParams + offset);
             assert(size == MODEL_DIM * MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.ln_2.bias", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].norm2.bias = (float*)(rawParams + offset);
             assert(size == MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.ln_2.weight", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].norm2.weight = (float*)(rawParams + offset);
             assert(size == MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.mlp.c_fc.bias", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].mlp.mlpFC.bias = (float*)(rawParams + offset);
             assert(size == 4 * MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.mlp.c_fc.weight", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].mlp.mlpFC.weight = (float*)(rawParams + offset);
             assert(size == MODEL_DIM * 4 * MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.mlp.c_proj.bias", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].mlp.mlpProj.bias = (float*)(rawParams + offset);
             assert(size == MODEL_DIM * sizeof(float));
 
             snprintf(label, sizeof(label), "h.%d.mlp.c_proj.weight", l);
             get_offset_and_size(jsonRaw, label, &offset, &size);
             modelParams->layers[l].mlp.mlpProj.weight = (float*)(rawParams + offset);
             assert(size == 4 * MODEL_DIM * MODEL_DIM * sizeof(float));
         }
         get_offset_and_size(jsonRaw, "ln_f.bias", &offset, &size);
         modelParams->finalNorm.bias = (float*)(rawParams + offset);
         assert(size == MODEL_DIM * sizeof(float));
 
         get_offset_and_size(jsonRaw, "ln_f.weight", &offset, &size);
         modelParams->finalNorm.weight = (float*)(rawParams + offset);
         assert(size == MODEL_DIM * sizeof(float));
 
         fclose(fParams);
     }
 
     /* === Choose Training or Inference Mode === */
     bool isTraining = VALIDATE_PERFORMANCE;  // Set true for training mode
 
     if (isTraining) {
         size_t inputSize = 64;  // Use a window of 64 tokens for training
         uint16_t* inputSequence = tokenData;
         uint16_t* expectedSequence = inputSequence + 1;
         memset(gradients, 0, sizeof(*gradients));
         process_transformer(modelParams, activations, inputSequence, inputSize, NULL, true, gradients, backwardActivations, expectedSequence);
     } else {
         size_t inputSize = (tokenCount < 64) ? tokenCount : 64;
         uint16_t* inputSequence = tokenData;
         for (size_t i = 0; i < inputSize; i++) {
             write(STDOUT_FILENO, decoder->raw + decoder->items[inputSequence[i]].offset,
                   decoder->items[inputSequence[i]].size);
         }
         for (int i = 0; i < 128; i++) {
             uint16_t outToken;
             process_transformer(modelParams, activations, inputSequence, inputSize, &outToken, false, NULL, NULL, NULL);
             write(STDOUT_FILENO, decoder->raw + decoder->items[outToken].offset,
                   decoder->items[outToken].size);
             inputSequence[inputSize++] = outToken;
         }
     }
 
     my_aligned_free(memoryBlock);
     return 0;
 }
 