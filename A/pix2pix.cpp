#include "pix2pix.h"

#include "util.h"

#include <pthread.h>
#include <string>
#include <map>
#include <cmath>

#define NUM_THREAD 16
#define BLOCK_SIZE 24
#define BLOCK_SIZE_K 64

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

  pthread_t thread[48];
  int params[48];

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

  double tz, ta, tb, tc, td, te;
  double t1, t2, t3, t4, t5;

  for (size_t img_idx = start; img_idx < end; ++img_idx) {
    // Pick 1 image out of num_image

    tz = get_time();

    get_one_image(input, one_image, img_idx);

    /*
     * Encoding phase
     */

    ta = get_time();

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
      if (f == 0) printf("\n HWC: %d %d %d\n", encoder_layer_input[i].shape[0], encoder_layer_input[i].shape[1], encoder_layer_input[i].shape[2]);
      t1 = get_time();
      leaky_relu(encoder_layer_input[i], encoder_layer_rectified[i], 0.2);
      t2 = get_time();
      conv2d(encoder_layer_rectified[i], filter, bias, encoder_layer_convolved[i]);
      t3 = get_time();
      batchnorm(encoder_layer_convolved[i], scale, offset, encoder_layer[i]);
      t4 = get_time();
      if (f == 0) printf("FUCK\n%.5f\n%.5f\n%.5f\nFUCK\n", t2 - t1, t3 - t2, t4 - t3);
    }

    tb = get_time();
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

    tc = get_time();
    // Convert values into [-1, 1] using tanh function
    elem_tanh(decoder_layer_convolved[1], decoder_layer[1]);
    
    td = get_time();

    // Put a image into output buffer
    postprocess_one_image(decoder_layer[1], output_buf, img_idx);

    te = get_time();
  }

  if (f == 0) {
    printf("\n%.5f\n%.5f\n%.5f\n%.5f\n%.5f\n", tz - ta, tb - ta, tc - tb, td - tc, te - td);
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
  register_weight(weights, &weight_buf, "generator/encoder_1/conv2d/kernel", {4, 4, 3, 64});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/beta", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/gamma", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_mean", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/batch_normalization/moving_variance", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/bias", {128});
  register_weight(weights, &weight_buf, "generator/encoder_2/conv2d/kernel", {4, 4, 64, 128});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/beta", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/gamma", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_mean", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/batch_normalization/moving_variance", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/bias", {256});
  register_weight(weights, &weight_buf, "generator/encoder_3/conv2d/kernel", {4, 4, 128, 256});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_4/conv2d/kernel", {4, 4, 256, 512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_5/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_6/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_7/conv2d/kernel", {4, 4, 512, 512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/beta", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/gamma", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_mean", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/batch_normalization/moving_variance", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/bias", {512});
  register_weight(weights, &weight_buf, "generator/encoder_8/conv2d/kernel", {4, 4, 512, 512});
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
  for (size_t i = 0; i < H * W * C; ++i) {
    float x = (input.buf[i] + 1) / 2 * 255;
    out[idx * (H * W * C) + i] = x < 0 ? 0 : (x > 255 ? 255 : x);
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
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
  const size_t stride = 2, pad = 1;
  size_t OH = H / stride, OW = W / stride;
  output.alloc_once({OH, OW, K});

  printf("H W C R S K: %d %d %d %d %d %d\n", H, W, C, R, S, K);

  for (size_t ohoh = 0; ohoh < OH; ohoh += BLOCK_SIZE) {
    size_t oh_end = ohoh + BLOCK_SIZE > OH ? OH : ohoh + BLOCK_SIZE;
  for (size_t owow = 0; owow < OW; owow += BLOCK_SIZE) {
    size_t ow_end = owow + BLOCK_SIZE > OW ? OW : owow + BLOCK_SIZE;
  for (size_t oh = ohoh; oh < oh_end; ++oh) {
    for (size_t ow = owow; ow < ow_end; ++ow) {
      for (size_t k = 0; k < K; k++) {
        output.buf[oh * OW * K + ow * K + k] = bias.buf[k];
      }
      for (size_t r = 0; r < R; ++r) {
        for (size_t s = 0; s < S; ++s) {
            // input (oh * stride - pad + r, ow * stride - pad + s, c)
          size_t ih = oh * stride - pad + r;
          size_t iw = ow * stride - pad + s;
          if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
          for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            size_t k_end = kk + BLOCK_SIZE > K ? K : kk + BLOCK_SIZE;
          for (size_t cc = 0; cc < C; cc += BLOCK_SIZE) {
            size_t c_end = cc + BLOCK_SIZE > C ? C : cc + BLOCK_SIZE;
          for (size_t k = kk; k < k_end; ++k) {
            float x = 0.0f;
            for (size_t c = cc; c < c_end; ++c) {
              float ii = input.buf[ih * W * C + iw * C + c]; // [ih][iw][c]
              // filter (r, s, c, k)
              float ff = filter.buf[r * S * C * K + s * C * K + c * K + k]; // [r][s][c][k]
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

  for (size_t ohoh = 0; ohoh < OH; ohoh += BLOCK_SIZE) {
    size_t oh_end = ohoh + BLOCK_SIZE > OH ? OH : ohoh + BLOCK_SIZE;
  for (size_t owow = 0; owow < OW; owow += BLOCK_SIZE) {
    size_t ow_end = owow + BLOCK_SIZE > OW ? OW : owow + BLOCK_SIZE;
  for (size_t oh = ohoh; oh < oh_end; ++oh) {
    for (size_t ow = owow; ow < ow_end; ++ow) {
      for (size_t k = 0; k < K; ++k) {
        float x = bias.buf[k];
        for (size_t r = 0; r < R; ++r) {
          for (size_t s = 0; s < S; ++s) {
              // input ((oh - r + pad) / stride, (ow - s + pad) / stride, c)
              //   where (oh - r + pad) % stride == 0 && (ow - s + pad) % stride == 0
            if ((oh - r + pad) % stride != 0 || (ow - s + pad) % stride != 0) continue;
            size_t ih = (oh - r + pad) / stride;
            size_t iw = (ow - s + pad) / stride;
            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
            for (size_t c = 0; c < C; ++c) {
              float ii = input.buf[ih * W * C + iw * C + c];
              // filter (r, s, k, c)
              float ff = filter.buf[r * S * K * C + s * K * C + k * C + c];
              x += ii * ff;
            }
          }
        }

        // output (oh, ow, k)
        output.buf[oh * OW * K + ow * K + k] = x;
      }
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
