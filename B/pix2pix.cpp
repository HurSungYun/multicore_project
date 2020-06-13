#include "pix2pix.h"

#include "util.h"

#include <string>
#include <map>
#include <cmath>

#include <CL/cl.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

#define BLOCK_SIZE 2

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


static cl_int err;
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel_conv2d, kernel_conv2d_transpose;

static cl_program create_and_build_program_with_source(cl_context context, cl_device_id device, const char *file_name);

void conv2d_gpu(Tensor input, Tensor filter, Tensor bias, Tensor &output);
void conv2d_transposed_gpu(Tensor input, Tensor filter, Tensor bias, Tensor &output);

static void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char*)malloc(sz);
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);
  free(buf);
}

void pix2pix_init() {
  /*
   * You can do input-independent and input-size-independent jobs here.
   * e.g., Getting OpenCL platform, Compiling OpenCL kernel, ...
   * Execution time of this function is not measured, so do as much as possible!
   */

  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);
  print_device_info(device);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  program = create_and_build_program_with_source(context, device, "kernel.cl");

  kernel_conv2d = clCreateKernel(program, "conv2d", &err);
  CHECK_ERROR(err);

  kernel_conv2d_transpose = clCreateKernel(program, "conv2d_transpose", &err);
  CHECK_ERROR(err);

  
}

void pix2pix(uint8_t *input_buf, float *weight_buf, uint8_t *output_buf, size_t num_image) {
  /*
   * !!!!!!!! Caution !!!!!!!!
   * In MPI program, all buffers and num_image are only given to rank 0 process.
   * You should manually:
   *   1. allocate buffers on others
   *   2. send inputs from rank 0 to others
   *   3. gather outputs from others to rank 0
   */

  auto weights = register_weights(weight_buf); // Memory allocated for weights
  auto input = preprocess(input_buf, num_image); // Memory allocated for input

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

  double t1, t2;

  for (size_t img_idx = 0; img_idx < num_image; ++img_idx) {
    // Pick 1 image out of num_image
    get_one_image(input, one_image, img_idx);

    /*
     * Encoding phase
     */

    // Encoder 1 : conv
    auto filter = weights["generator/encoder_1/conv2d/kernel"];
    auto bias = weights["generator/encoder_1/conv2d/bias"];
    conv2d_gpu(one_image, filter, bias, encoder_layer[1]);

    for (int i = 2; i <= 8; ++i) {
      // Encoder i : leaky_relu => conv2d => batchnorm
      auto scope = "generator/encoder_" + std::to_string(i);
      auto filter = weights[scope + "/conv2d/kernel"];
      auto bias = weights[scope + "/conv2d/bias"];
      auto scale = weights[scope + "/batch_normalization/gamma"];
      auto offset = weights[scope + "/batch_normalization/beta"];

      encoder_layer_input[i] = encoder_layer[i - 1];
      leaky_relu(encoder_layer_input[i], encoder_layer_rectified[i], 0.2);
      //conv2d(encoder_layer_rectified[i], filter, bias, encoder_layer_convolved[i]);
      t1 = get_time();
      conv2d_gpu(encoder_layer_rectified[i], filter, bias, encoder_layer_convolved[i]);
      t2 = get_time();
      printf("\nFUCK\n%.5f\n", t2 - t1);
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
      //conv2d_transposed(decoder_layer_rectified[i], filter, bias, decoder_layer_convolved[i]);
      conv2d_transposed_gpu(decoder_layer_rectified[i], filter, bias, decoder_layer_convolved[i]);

      // Last decoder does not have batchnorm
      if (i == 1) break;
      batchnorm(decoder_layer_convolved[i], scale, offset, decoder_layer[i]);
    }
    // Convert values into [-1, 1] using tanh function
    elem_tanh(decoder_layer_convolved[1], decoder_layer[1]);

    // Put a image into output buffer
    postprocess_one_image(decoder_layer[1], output_buf, img_idx);
  }
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

  for (size_t oh = 0; oh < OH; ++oh) {
    for (size_t ow = 0; ow < OW; ++ow) {
      for (size_t k = 0; k < K; ++k) {
        float x = bias.buf[k];
        for (size_t c = 0; c < C; ++c) {
          for (size_t r = 0; r < R; ++r) {
            for (size_t s = 0; s < S; ++s) {
              // input (oh * stride - pad + r, ow * stride - pad + s, c)
              size_t ih = oh * stride - pad + r;
              size_t iw = ow * stride - pad + s;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
              float ii = input.buf[ih * W * C + iw * C + c];
              // filter (r, s, c, k)
              float ff = filter.buf[r * S * C * K + s * C * K + c * K + k];
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

void conv2d_gpu(Tensor input, Tensor filter, Tensor bias, Tensor &output) {
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[3];
  const size_t stride = 2, pad = 1;
  size_t OH = H / stride, OW = W / stride;
  output.alloc_once({OH, OW, K});

  printf("\nH W C K: %d %d %d %d\n", H, W, C, K);

  if (R != 4 || S != 4) {
      printf("\nFUCK %d %d\n", R, S);
  }

  double t1, t2, t3, t4;

  t1 = get_time();

  size_t gws[2] = {OH, OW}, lws[2] = {BLOCK_SIZE, BLOCK_SIZE};
  for (int i = 0; i < 2; i++) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  cl_mem input_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * C * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  cl_mem filter_d = clCreateBuffer(context, CL_MEM_READ_WRITE, R * S * C * K * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  cl_mem bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  cl_mem output_d = clCreateBuffer(context, CL_MEM_READ_WRITE, OH * OW * K * sizeof(float), NULL, &err);
  CHECK_ERROR(err);


  err = clSetKernelArg(kernel_conv2d, 0, sizeof(cl_mem), &input_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d, 1, sizeof(cl_mem), &filter_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d, 2, sizeof(cl_mem), &bias_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d, 3, sizeof(cl_mem), &output_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d, 4, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d, 5, sizeof(int), &W);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d, 6, sizeof(int), &C);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d, 7, sizeof(int), &R);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d, 8, sizeof(int), &S);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d, 9, sizeof(int), &K);
  CHECK_ERROR(err);

  t2 = get_time();
  
  err = clEnqueueWriteBuffer(queue, input_d, CL_TRUE, 0, H * W * C * sizeof(float), input.buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  err = clEnqueueWriteBuffer(queue, filter_d, CL_TRUE, 0, R * S * C * K * sizeof(float), filter.buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  err = clEnqueueWriteBuffer(queue, bias_d, CL_TRUE, 0, K * sizeof(float), bias.buf, 0, NULL, NULL);
  CHECK_ERROR(err);

  t3 = get_time();

  err = clEnqueueNDRangeKernel(queue, kernel_conv2d, 2, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);

  err = clEnqueueReadBuffer(queue, output_d, CL_TRUE, 0, OH * OW * K * sizeof(float), output.buf, 0, NULL, NULL);
  CHECK_ERROR(err);

  t4 = get_time();

  printf("\nKILL\n%.5f\n%.5f\n%.5f\nKILL\n", t2 - t1, t3 - t2, t4 - t3);
  
  err = clFinish(queue);
  CHECK_ERROR(err);
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

  for (size_t oh = 0; oh < OH; ++oh) {
    for (size_t ow = 0; ow < OW; ++ow) {
      for (size_t k = 0; k < K; ++k) {
        float x = bias.buf[k];
        for (size_t r = 0; r < R; ++r) {
          for (size_t s = 0; s < S; ++s) {
            for (size_t c = 0; c < C; ++c) {
              // input ((oh - r + pad) / stride, (ow - s + pad) / stride, c)
              //   where (oh - r + pad) % stride == 0 && (ow - s + pad) % stride == 0
              if ((oh - r + pad) % stride != 0 || (ow - s + pad) % stride != 0) continue;
              size_t ih = (oh - r + pad) / stride;
              size_t iw = (ow - s + pad) / stride;
              if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
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

void conv2d_transposed_gpu(Tensor input, Tensor filter, Tensor bias, Tensor &output) {
  size_t H = input.shape[0], W = input.shape[1], C = input.shape[2];
  size_t R = filter.shape[0], S = filter.shape[1], K = filter.shape[2];
  const size_t stride = 2, pad = 1;
  size_t OH = H * stride, OW = W * stride;
  output.alloc_once({OH, OW, K});

  printf("\nH W C K: %d %d %d %d\n", H, W, C, K);

  if (R != 4 || S != 4) {
      printf("\nFUCK %d %d\n", R, S);
  }

  double t1, t2, t3, t4;

  t1 = get_time();

  size_t gws[2] = {OH, OW}, lws[2] = {BLOCK_SIZE, BLOCK_SIZE};
  for (int i = 0; i < 2; i++) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }
  cl_mem input_d = clCreateBuffer(context, CL_MEM_READ_WRITE, H * W * C * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  cl_mem filter_d = clCreateBuffer(context, CL_MEM_READ_WRITE, R * S * C * K * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  cl_mem bias_d = clCreateBuffer(context, CL_MEM_READ_WRITE, K * sizeof(float), NULL, &err);
  CHECK_ERROR(err);
  cl_mem output_d = clCreateBuffer(context, CL_MEM_READ_WRITE, OH * OW * K * sizeof(float), NULL, &err);
  CHECK_ERROR(err);

  
  err = clSetKernelArg(kernel_conv2d_transpose, 0, sizeof(cl_mem), &input_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d_transpose, 1, sizeof(cl_mem), &filter_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d_transpose, 2, sizeof(cl_mem), &bias_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d_transpose, 3, sizeof(cl_mem), &output_d);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d_transpose, 4, sizeof(int), &H);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d_transpose, 5, sizeof(int), &W);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d_transpose, 6, sizeof(int), &C);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d_transpose, 7, sizeof(int), &R);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d_transpose, 8, sizeof(int), &S);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel_conv2d_transpose, 9, sizeof(int), &K);
  CHECK_ERROR(err);

  t2 = get_time();
  
  err = clEnqueueWriteBuffer(queue, input_d, CL_TRUE, 0, H * W * C * sizeof(float), input.buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  err = clEnqueueWriteBuffer(queue, filter_d, CL_TRUE, 0, R * S * C * K * sizeof(float), filter.buf, 0, NULL, NULL);
  CHECK_ERROR(err);
  err = clEnqueueWriteBuffer(queue, bias_d, CL_TRUE, 0, K * sizeof(float), bias.buf, 0, NULL, NULL);
  CHECK_ERROR(err);

  t3 = get_time();

  err = clEnqueueNDRangeKernel(queue, kernel_conv2d_transpose, 2, NULL, gws, lws, 0, NULL, NULL);
  CHECK_ERROR(err);

  err = clEnqueueReadBuffer(queue, output_d, CL_TRUE, 0, OH * OW * K * sizeof(float), output.buf, 0, NULL, NULL);
  CHECK_ERROR(err);

  t4 = get_time();

  printf("\nTRANSPOSE\n%.5f\n%.5f\n%.5f\nTRANSPOSE\n", t2 - t1, t3 - t2, t4 - t3);
  
  err = clFinish(queue);
  CHECK_ERROR(err);

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

static cl_program create_and_build_program_with_source(cl_context context, cl_device_id device, const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char*)malloc(source_size + 1);
  fread(source_code, sizeof(char), source_size, file);
  source_code[source_size] = '\0';
  fclose(file);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_code, &source_size, &err);
  CHECK_ERROR(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
    char *log = (char*)malloc(log_size + 1);
    CHECK_ERROR(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);
  return program;
}

