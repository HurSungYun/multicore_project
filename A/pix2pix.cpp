#include "pix2pix.h"

#include "util.h"

#include <pthread.h>
#include <string>
#include <map>
#include <cmath>
#include <immintrin.h>

#define NUM_THREAD 32
#define BLOCK_SIZE_K 32
#define BLOCK_SIZE_C 128

#define LEAKY_RELU_ALPHA 0.2

class Tensor {
public:
  Tensor();
  Tensor(float *buf_, std::vector<size_t> shape_);
  void alloc_once(std::vector<size_t> shape_);
  void set_sz();

  // For real world application, one should use smart pointer to prevent possible memory leak.
  // However, we just use raw pointer for simplicity. Memory will be freed anyway when process exits.
  float* buf;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., [[1, 2, 3], [4, 5, 6]] => shape = [2, 3]
  std::vector<size_t> shape;

  // Size of tensor; product of all dimensions
  size_t sz;
};

// Helpers
static void register_weight(std::map<std::string, Tensor>& weights, float* (*buf), std::string name, std::vector<size_t> shape);
static std::map<std::string, Tensor> register_weights(float* weight_buf);
static Tensor preprocess(uint8_t *in, size_t num_image);
static void postprocess_one_image(Tensor input, uint8_t *out, size_t idx);
static void get_one_image(Tensor input, Tensor &output, size_t idx);

// Operators
static void conv2d(Tensor input, Tensor filter, Tensor bias, Tensor &output);
static void conv2d_transposed(Tensor input, Tensor filter, Tensor bias, Tensor &output);
static void leaky_relu(Tensor input, Tensor &output, float alpha);
static void relu(Tensor input, Tensor &output);
static void batchnorm(Tensor input, Tensor scale, Tensor offset, Tensor &output);
static void concat(Tensor input0, Tensor input1, Tensor &output);
static void elem_tanh(Tensor input, Tensor &output);

void pix2pix_init() {
  /*
   * You can do input-independent and input-size-independent jobs here.
   * e.g., Getting OpenCL platform, Compiling OpenCL kernel, ...
   * Execution time of this function is not measured, so do as much as possible!
   */
}

static uint8_t *input_buf;
static float *weight_buf;
static uint8_t *output_buf;
size_t num_image;

std::map<std::string, Tensor> weights;
Tensor input;

static void* pix2pix_thread(void *data);

void pix2pix(uint8_t *_input_buf, float *_weight_buf, uint8_t *_output_buf, size_t _num_image) {
  /*
   * !!!!!!!! Caution !!!!!!!!
   * In MPI program, all buffers and num_image are only given to rank 0 process.
   * You should manually:
   *   1. allocate buffers on others
   *   2. send inputs from rank 0 to others
   *   3. gather outputs from others to rank 0
   */
  input_buf = _input_buf;
  weight_buf = _weight_buf;
  output_buf = _output_buf;
  num_image = _num_image;

  weights = register_weights(weight_buf); // Memory allocated for weights
  input = preprocess(input_buf, num_image); // Memory allocated for input

  pthread_t thread[NUM_THREAD];
  int params[NUM_THREAD];

  for (int i = 0; i < NUM_THREAD; i++) {
    params[i] = i;
    pthread_create(&thread[i], NULL, pix2pix_thread, &params[i]);
  }

  for (int i = 0; i < NUM_THREAD; i++) {
    pthread_join(thread[i], NULL);
  }
}

static void* pix2pix_thread(void *data) {

  // Declare feature maps
  // Memory for feature maps are allocated when they are written first time using Tensor::alloc_once(...)
  Tensor one_image;
  Tensor encoder_layer_input[9];
  Tensor encoder_layer_rectified[9];
  Tensor encoder_layer_convolved[9];
  Tensor encoder_layer[9];
  Tensor decoder_layer_input[9];
  Tensor decoder_layer_rectified[9];
  Tensor decoder_layer_convolved[9];
  Tensor decoder_layer[9];
  
  size_t f = *((int *)data);
  size_t quota = (num_image + NUM_THREAD - 1) / NUM_THREAD;
  size_t start = f * quota;
  size_t end = (f + 1) * quota > num_image ? num_image : (f + 1) * quota;

  for (size_t img_idx = start; img_idx < end; ++img_idx) {
    // Pick 1 image out of num_image

    get_one_image(input, one_image, img_idx);

    /*
     * Encoding phase
     */

    // Encoder 1 : conv
    auto filter = weights["generator/encoder_1/conv2d/kernel"];
    auto bias = weights["generator/encoder_1/conv2d/bias"];
    conv2d(one_image, filter, bias, encoder_layer[1]);


    for (int i = 2; i <= 8; ++i) {
      // Encoder i : leaky_relu => conv2d => batchnorm
      auto scope = "generator/encoder_" + std::to_string(i);
      auto filter = weights[scope + "/conv2d/kernel"];
      auto bias = weights[scope + "/conv2d/bias"];
      auto scale = weights[scope + "/batch_normalization/gamma"];
      auto offset = weights[scope + "/batch_normalization/beta"];
      encoder_layer_input[i] = encoder_layer[i - 1]; // <- dependence
      leaky_relu(encoder_layer_input[i], encoder_layer_rectified[i], 0.2);
      conv2d(encoder_layer_rectified[i], filter, bias, encoder_layer_convolved[i]);
      batchnorm(encoder_layer_convolved[i], scale, offset, encoder_layer[i]);
    }

    /*
     * Decoding phase
     */

    for (int i = 8; i >= 1; --i) {
      // Decoder i : relu => conv2d_transposed => batchnorm
      auto scope = "generator/decoder_" + std::to_string(i);
      auto filter = weights[scope + "/conv2d_transpose/kernel"];
      auto bias = weights[scope + "/conv2d_transpose/bias"];
      auto scale = weights[scope + "/batch_normalization/gamma"];
      auto offset = weights[scope + "/batch_normalization/beta"];
      if (i == 8) {
        // For decoder 8, input is last layer of encoder
        decoder_layer_input[i] = encoder_layer[8];
      } else {
        // For other decoder, input is concatenation of previous layer and corresponding encoder layer
        concat(decoder_layer[i + 1], encoder_layer[i], decoder_layer_input[i]);
      }
      relu(decoder_layer_input[i], decoder_layer_rectified[i]);
      conv2d_transposed(decoder_layer_rectified[i], filter, bias, decoder_layer_convolved[i]);

      // Last decoder does not have batchnorm
      if (i == 1) break;
      batchnorm(decoder_layer_convolved[i], scale, offset, decoder_layer[i]);
    }

    // Convert values into [-1, 1] using tanh function
    elem_tanh(decoder_layer_convolved[1], decoder_layer[1]);
    
    // Put a image into output buffer
    postprocess_one_image(decoder_layer[1], output_buf, img_idx);
  }

  return NULL;
}

Tensor::Tensor() : buf(NULL) {}

// If buf is given, use it. If not, allocate new one.
Tensor::Tensor(float *buf_, std::vector<size_t> shape_) : buf(buf_), shape(shape_) {
  set_sz();
  if (buf == NULL) {
    buf = (float*)malloc(sz * sizeof(float));
  }
}

// If buf is not allocated, allocate new one.
void Tensor::alloc_once(std::vector<size_t> shape_) {
  if (buf == NULL) {
    shape = shape_;
    set_sz();
    buf = (float*)malloc(sz * sizeof(float));
  }
}

void Tensor::set_sz() {
  sz = 1;
  for (auto x : shape) {
    sz *= x;
  }
}

// Make a new tensor from buffer and put the tensor into map. Advance buffer pointer by size.
void register_weight(std::map<std::string, Tensor>& weights, float* (*buf), std::string name, std::vector<size_t> shape) {
  Tensor tensor(*buf, shape);
  weights[name] = tensor;
  *buf += tensor.sz;
}

void register_weight_trans(std::map<std::string, Tensor>& weights, float* (*buf), std::string name, std::vector<size_t> shape) {
  Tensor t;
  size_t R = shape[0], S = shape[1], C = shape[2], K = shape[3];
  t.alloc_once({R, S, K, C});

  for (size_t r = 0; r < R; r++) {
    for (size_t s = 0; s < S; s++) {
      for (size_t c = 0; c < C; c++) {
        for (size_t k = 0; k < K; k++) {
            t.buf[r * S * K * C + s * K * C + k * C + c] = (*buf)[r * S * C * K + s * C * K + c * K + k];
        }
      }
    }
  }

  weights[name] = t;
  *buf += t.sz;
}

// Put all predefined weights into map. Order should not be changed.
std::map<std::string, Tensor> register_weights(float* weight_buf) {
  std::map<std::string, Tensor> weights;
  // auto generated
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/bias", {3});
  register_weight(weights, &weight_buf, "generator/decoder_1/conv2d_transpose/kernel", {4, 4, 3, 128});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/beta", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/gamma", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_mean", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/batch_normalization/moving_variance", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/bias", {64});
  register_weight(weights, &weight_buf, "generator/decoder_2/conv2d_transpose/kernel", {4, 4, 64, 256});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/bias", {128});
  register_weight(weights, &weight_buf, "generator/decoder_3/conv2d_transpose/kernel", {4, 4, 128, 512});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/bias", {256});
  register_weight(weights, &weight_buf, "generator/decoder_4/conv2d_transpose/kernel", {4, 4, 256, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_5/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_6/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_7/conv2d_transpose/kernel", {4, 4, 512, 1024});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/bias", {512});
  register_weight(weights, &weight_buf, "generator/decoder_8/conv2d_transpose/kernel", {4, 4, 512, 512});


  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/bias", {64});
  register_weight_trans(weights, &weight_buf, "generator/encoder_1/conv2d/kernel", {4, 4, 3, 64});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/bias", {128});
  register_weight_trans(weights, &weight_buf, "generator/encoder_2/conv2d/kernel", {4, 4, 64, 128});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/bias", {256});
  register_weight_trans(weights, &weight_buf, "generator/encoder_3/conv2d/kernel", {4, 4, 128, 256});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/bias", {512});
  register_weight_trans(weights, &weight_buf, "generator/encoder_4/conv2d/kernel", {4, 4, 256, 512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/bias", {512});
  register_weight_trans(weights, &weight_buf, "generator/encoder_5/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/bias", {512});
  register_weight_trans(weights, &weight_buf, "generator/encoder_6/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/bias", {512});
  register_weight_trans(weights, &weight_buf, "generator/encoder_7/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/bias", {512});
  register_weight_trans(weights, &weight_buf, "generator/encoder_8/conv2d/kernel", {4, 4, 512, 512});
  return weights;
}

// Convert 8-bit depth images (value range [0, 255]) into floating-point ones (value range [-1, 1])
Tensor preprocess(uint8_t *in, size_t num_image) {
  Tensor out(NULL, {num_image, 256, 256, 3});
  for (size_t i = 0; i < out.sz; ++i) {
    out.buf[i] = in[i] / 255.0f * 2 - 1;
  }
  return out;
}

// Inverse of preprocess
void postprocess_one_image(Tensor input, uint8_t *out, size_t idx) {
  // input shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t idx_offset = idx * (H * W * C);
  for (size_t i = 0; i < H * W * C; ++i) {
    float x = (input.buf[i] + 1) / 2 * 255;
    out[idx_offset + i] = x < 0 ? 0 : (x > 255 ? 255 : x);
  }
}

// Pick single image from images
void get_one_image(Tensor input, Tensor &output, size_t idx) {
  // input shape = (num_image, height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[1], W = input.shape[2], C = input.shape[3];
  output.alloc_once({H, W, C});
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = input.buf[idx * H * W * C + i];
  }
}

// Convolution (2-dimension, stride = 2, pad = 1)
void conv2d(Tensor input, Tensor filter, Tensor bias, Tensor &output) {
  // input shape = (in_height, in_width, in_channels)
  // filter shape = (filter_height, filter_width, in_channels, output_channels)
  // bias shape = (output_channels)
  // output shape = (in_height / stride, in_width / stride, output_channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[2];
  const size_t stride = 2, pad = 1;
  size_t OH = H / stride, OW = W / stride;
  output.alloc_once({OH, OW, K});

  for (size_t oh = 0; oh < OH; oh++) {
    for (size_t ow = 0; ow < OW; ow++) {
      for (size_t k = 0; k < K; k++) {
        output.buf[oh * OW * K + ow * K + k] = bias.buf[k];
      }
    }
  }
  for (size_t kk = 0; kk < K; kk += BLOCK_SIZE_K) {
    size_t k_end = kk + BLOCK_SIZE_K > K ? K : kk + BLOCK_SIZE_K;
  for (size_t cc = 0; cc < C; cc += BLOCK_SIZE_C) {
    size_t c_end = cc + BLOCK_SIZE_C > C ? C : cc + BLOCK_SIZE_C;


  for (size_t r = 0; r < R; ++r) {
  for (size_t s = 0; s < S; ++s) {
  for (size_t oh = 0; oh < OH; ++oh) {
    size_t ih = oh * stride - pad + r;
    if (ih < 0 || ih >= H) continue;
    for (size_t ow = 0; ow < OW; ++ow) {
      size_t iw = ow * stride - pad + s;
      if (iw < 0 || iw >= W) continue;
          for (size_t k = kk; k < k_end; ++k) {
            if (c_end - cc == BLOCK_SIZE_C) { // BLOCK_SIZE_C == 32
              float buf_input[BLOCK_SIZE_C];
              float buf_filter[BLOCK_SIZE_C];
              for (size_t c = cc; c < c_end; c++) {
                buf_input[c - cc] = (float) input.buf[ih * W * C + iw * C + c];
                buf_filter[c - cc] = (float) filter.buf[r * S * K * C + s * K * C + k * C + c];
              }

              __m256 temp = {0.0f, };
              
              __m256 ii1 = _mm256_loadu_ps(&buf_input[0]); // uint8_t
              __m256 ff1 = _mm256_loadu_ps(&buf_filter[0]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii1, ff1), temp);
              
              __m256 ii2 = _mm256_loadu_ps(&buf_input[8]);
              __m256 ff2 = _mm256_loadu_ps(&buf_filter[8]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii2, ff2), temp);
              
              __m256 ii3 = _mm256_loadu_ps(&buf_input[16]);
              __m256 ff3 = _mm256_loadu_ps(&buf_filter[16]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii3, ff3), temp);
              
              __m256 ii4 = _mm256_loadu_ps(&buf_input[24]);
              __m256 ff4 = _mm256_loadu_ps(&buf_filter[24]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii4, ff4), temp);
              
              __m256 ii5 = _mm256_loadu_ps(&buf_input[32]);
              __m256 ff5 = _mm256_loadu_ps(&buf_filter[32]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii5, ff5), temp);

              __m256 ii6 = _mm256_loadu_ps(&buf_input[40]);
              __m256 ff6 = _mm256_loadu_ps(&buf_filter[40]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii6, ff6), temp);

              __m256 ii7 = _mm256_loadu_ps(&buf_input[48]);
              __m256 ff7 = _mm256_loadu_ps(&buf_filter[48]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii7, ff7), temp);

              __m256 ii8 = _mm256_loadu_ps(&buf_input[56]);
              __m256 ff8 = _mm256_loadu_ps(&buf_filter[56]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii8, ff8), temp);

              __m256 ii9 = _mm256_loadu_ps(&buf_input[64]); // uint8_t
              __m256 ff9 = _mm256_loadu_ps(&buf_filter[64]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii9, ff9), temp);
              
              __m256 ii10 = _mm256_loadu_ps(&buf_input[72]);
              __m256 ff10 = _mm256_loadu_ps(&buf_filter[72]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii10, ff10), temp);
              
              __m256 ii11 = _mm256_loadu_ps(&buf_input[80]);
              __m256 ff11 = _mm256_loadu_ps(&buf_filter[80]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii11, ff11), temp);
              
              __m256 ii12 = _mm256_loadu_ps(&buf_input[88]);
              __m256 ff12 = _mm256_loadu_ps(&buf_filter[88]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii12, ff12), temp);
              
              __m256 ii13 = _mm256_loadu_ps(&buf_input[96]);
              __m256 ff13 = _mm256_loadu_ps(&buf_filter[96]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii13, ff13), temp);

              __m256 ii14 = _mm256_loadu_ps(&buf_input[104]);
              __m256 ff14 = _mm256_loadu_ps(&buf_filter[104]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii14, ff14), temp);

              __m256 ii15 = _mm256_loadu_ps(&buf_input[112]);
              __m256 ff15 = _mm256_loadu_ps(&buf_filter[112]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii15, ff15), temp);

              __m256 ii16 = _mm256_loadu_ps(&buf_input[120]);
              __m256 ff16 = _mm256_loadu_ps(&buf_filter[120]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii16, ff16), temp);



              output.buf[oh * OW * K + ow * K + k] += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
 

            } else {
              float x = 0.0f;
              for (size_t c = cc; c < c_end; ++c) {
                float ii = input.buf[ih * W * C + iw * C + c]; // [ih][iw][c]
                // filter (r, s, c, k)
                float ff = filter.buf[r * S * K * C + s * K * C + k * C + c]; // [r][s][c][k]
                x += ii * ff;
              }
              output.buf[oh * OW * K + ow * K + k] += x;
            }
          }
    }
  }
  }
  }
  }
  }
}

// Transposed convolution (2-dimension, stride = 2, pad = 1)
void conv2d_transposed(Tensor input, Tensor filter, Tensor bias, Tensor &output) {
  // input shape = (in_height, in_width, in_channels)
  // filter shape = (filter_height, filter_width, output_channels, in_channels)
  // bias shape = (output_channels)
  // output shape = (in_height * stride, in_width * stride, output_channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[2];
  // assume stride 2, pad 1
  const size_t stride = 2, pad = 1;
  size_t OH = H * stride, OW = W * stride;
  output.alloc_once({OH, OW, K});
  
  for (size_t oh = 0; oh < OH; oh++) {
    for (size_t ow = 0; ow < OW; ow++) {
      for (size_t k = 0; k < K; k++) {
        output.buf[oh * OW * K + ow * K + k] = bias.buf[k];
      }
    }
  }

  for (size_t kk = 0; kk < K; kk += BLOCK_SIZE_K) {
    size_t k_end = kk + BLOCK_SIZE_K > K ? K : kk + BLOCK_SIZE_K;
  for (size_t cc = 0; cc < C; cc += BLOCK_SIZE_C) {
    size_t c_end = cc + BLOCK_SIZE_C > C ? C : cc + BLOCK_SIZE_C;

  for (size_t r = 0; r < R; r++) {
  for (size_t s = 0; s < S; s++) {
//  for (size_t ihih = 0; ihih < H; ihih += BLOCK_SIZE) {
//    size_t ih_end = ihih + BLOCK_SIZE > H ? H : ihih + BLOCK_SIZE;
//  for (size_t iwiw = 0; iwiw < W; iwiw += BLOCK_SIZE) {
//    size_t iw_end = iwiw + BLOCK_SIZE > W ? W : iwiw + BLOCK_SIZE;
  for (size_t ih = 0; ih < H; ih++) {
    size_t oh = ih * stride - pad + r;
    if (oh < 0 || oh >= OH) continue;
    for (size_t iw = 0; iw < W; iw++) {
          size_t ow = iw * stride - pad + s;
          if (ow < 0 || ow >= OW) continue;

          for (size_t k = kk; k < k_end; ++k) {
            float x = 0.0f;
            if (c_end - cc == BLOCK_SIZE_C) { // BLOCK_SIZE_C == 32
                float buf[BLOCK_SIZE_C];
                for (size_t c = cc; c < c_end; c++) {
                  buf[c - cc] = (float) input.buf[ih * W * C + iw * C + c];
                }

              __m256 temp = {0.0f, };
              
              __m256 ii1 = _mm256_loadu_ps(&buf[0]); // uint8_t
              __m256 ff1 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii1, ff1), temp);
              
              __m256 ii2 = _mm256_loadu_ps(&buf[8]);
              __m256 ff2 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 8]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii2, ff2), temp);
              
              __m256 ii3 = _mm256_loadu_ps(&buf[16]);
              __m256 ff3 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 16]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii3, ff3), temp);
              
              __m256 ii4 = _mm256_loadu_ps(&buf[24]);
              __m256 ff4 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 24]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii4, ff4), temp);
              
              __m256 ii5 = _mm256_loadu_ps(&buf[32]);
              __m256 ff5 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 32]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii5, ff5), temp);

              __m256 ii6 = _mm256_loadu_ps(&buf[40]);
              __m256 ff6 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 40]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii6, ff6), temp);

              __m256 ii7 = _mm256_loadu_ps(&buf[48]);
              __m256 ff7 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 48]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii7, ff7), temp);

              __m256 ii8 = _mm256_loadu_ps(&buf[56]);
              __m256 ff8 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 56]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii8, ff8), temp);

              __m256 ii9 = _mm256_loadu_ps(&buf[64]); // uint8_t
              __m256 ff9 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 64]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii9, ff9), temp);
              
              __m256 ii10 = _mm256_loadu_ps(&buf[72]);
              __m256 ff10 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 72]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii10, ff10), temp);
              
              __m256 ii11 = _mm256_loadu_ps(&buf[80]);
              __m256 ff11 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 80]);
              
              temp = _mm256_add_ps(_mm256_mul_ps(ii11, ff11), temp);
              
              __m256 ii12 = _mm256_loadu_ps(&buf[88]);
              __m256 ff12 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 88]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii12, ff12), temp);
              
              __m256 ii13 = _mm256_loadu_ps(&buf[96]);
              __m256 ff13 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 96]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii13, ff13), temp);

              __m256 ii14 = _mm256_loadu_ps(&buf[104]);
              __m256 ff14 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 104]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii14, ff14), temp);

              __m256 ii15 = _mm256_loadu_ps(&buf[112]);
              __m256 ff15 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 112]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii15, ff15), temp);

              __m256 ii16 = _mm256_loadu_ps(&buf[120]);
              __m256 ff16 = _mm256_loadu_ps(&filter.buf[r * S * K * C + s * K * C + k * C + cc + 120]);

              temp = _mm256_add_ps(_mm256_mul_ps(ii16, ff16), temp);



              output.buf[oh * OW * K + ow * K + k] += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            } else {
              for (size_t c = cc; c < c_end; ++c) {
                float ii = input.buf[ih * W * C + iw * C + c];
                // filter (r, s, k, c)
                float ff = filter.buf[r * S * K * C + s * K * C + k * C + c];
                x += ii * ff;
              }
              output.buf[oh * OW * K + ow * K + k] += x;
            }
          }
    }
  }
//  }
//  }
  }
  }
  }
  }

}

// Leaky ReLU
void leaky_relu(Tensor input, Tensor &output, float alpha) {
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = input.buf[i] >= 0 ? input.buf[i] : alpha * input.buf[i];
  }
}

// ReLU
void relu(Tensor input, Tensor &output) {
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = input.buf[i] >= 0 ? input.buf[i] : 0;
  }
}

// Batch normalization (channel-wise)
void batchnorm(Tensor input, Tensor scale, Tensor offset, Tensor &output) {
  // input shape = (height, width, channels)
  // scale shape = (channels)
  // offset shape = (channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  for (size_t c = 0; c < C; ++c) {
    float sum = 0;
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        float ii = input.buf[h * W * C + w * C + c];
        sum += ii;
      }
    }
    float mean = sum / (H * W);

    float sqsum = 0;
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        float ii = input.buf[h * W * C + w * C + c];
        sqsum += (ii - mean) * (ii - mean);
      }
    }
    float variance = sqsum / (H * W);

    const float epsilon = 1e-5;
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        size_t idx = h * W * C + w * C + c;
        output.buf[idx] = offset.buf[c] + (input.buf[idx] - mean) * scale.buf[c] / sqrtf(variance + epsilon);
      }
    }
  }
}

// Concatenation (along channel dimension)
void concat(Tensor input0, Tensor input1, Tensor &output) {
  // input0 shape = (height, width, channels0)
  // input1 shape = (height, width, channels1)
  // output shape = (height, width, channels0 + channels1)
  size_t H = input0.shape[0], W = input0.shape[1], C0 = input0.shape[2];
  size_t C1 = input1.shape[2];
  output.alloc_once({H, W, C0 + C1});
  for (size_t h = 0; h < H; ++h) {
    for (size_t w = 0; w < W; ++w) {
      for (size_t c = 0; c < C0; ++c) {
        output.buf[h * W * (C0 + C1) + w * (C0 + C1) + c] = input0.buf[h * W * C0 + w * C0 + c];
      }
      for (size_t c = 0; c < C1; ++c) {
        output.buf[h * W * (C0 + C1) + w * (C0 + C1) + (C0 + c)] = input1.buf[h * W * C1 + w * C1 + c];
      }
    }
  }
}

// Elementwise tanh
void elem_tanh(Tensor input, Tensor &output) {
  // input shape = (height, width, channels)
  // output shape = (height, width, channels)
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  output.alloc_once({H, W, C});
  for (size_t i = 0; i < H * W * C; ++i) {
    output.buf[i] = tanhf(input.buf[i]);
  }
}
