#define BLOCK_SIZE 2
#define STRIDE 2
#define PAD 1
#define CACHE_SIZE 24

__kernel void conv2d(__global float *input, __global float *filter, __global float *bias, __global float *output, int H, int W, int C, int R, int S, int K) {
  int local_oh = get_local_id(0);
  int local_ow = get_local_id(1);
  int local_k = get_local_id(2);
  int oh = get_global_id(0);
  int ow = get_global_id(1);
  int k = get_group_id(2) * CACHE_SIZE + local_k;
  
  int local_c = local_k;

  int OH = H / STRIDE;
  int OW = W / STRIDE;

  __local float local_input[CACHE_SIZE];

  float x = k < K ? bias[k] : 0;

  for (int r = 0; r < R; r++) {
    for (int s = 0; s < S; s++) {
      int iw = ow * STRIDE - PAD + s;
      int ih = oh * STRIDE - PAD + r;
      int tiles = (C + CACHE_SIZE - 1) / CACHE_SIZE;
      for (int t = 0; t < tiles; t++) {
        int c_base = t * CACHE_SIZE;
        int c = c_base + local_c;
      if (ih < 0 || ih >= H || iw < 0 || iw >= W) { 
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_LOCAL_MEM_FENCE);
      } else {
        if (c < C) {
          local_input[local_c] = input[ih * W * C + iw * C + c_base + local_c];
        } else {
          local_input[local_c] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (k < K) {
          int c_max = c_base + CACHE_SIZE > C ? C - c_base : CACHE_SIZE;

          for (int c_offset = 0; c_offset < c_max; c_offset++) {
            x += local_input[c_offset] * filter[r * S * C * K + s * C * K + (c_base + c_offset) * K + k];
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      }
    }
  }
  if (ow < OW && oh < OH && k < K) output[oh * OW * K + ow * K + k] = x;

/*
    float x = bias[k];
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        for (int c = 0; c < C; ++c) {
          int iw = ow * STRIDE - PAD + s;
          int ih = oh * STRIDE - PAD + r;
          if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
          float ii = input[ih * W * C + iw * C + c];
          float ff = filter[r * S * C * K + s * C * K + c * K + k];
          x += ii * ff;
        }
      }
    }
    if (ow < OW && oh < OH) output[oh * OW * K + ow * K + k] = x;
*/
}


__kernel void conv2d_transpose(__global float *input, __global float *filter, __global float *bias, __global float *output, int H, int W, int C, int R, int S, int K) {
  int local_h = get_local_id(0);
  int local_w = get_local_id(1);
  int oh = get_group_id(0) * BLOCK_SIZE + get_local_id(0);
  int ow = get_group_id(1) * BLOCK_SIZE + get_local_id(1);

  int OH = H * STRIDE;
  int OW = W * STRIDE;

  for (size_t k = 0; k < K; ++k) {
    float x = bias[k];
    for (size_t r = 0; r < R; ++r) {
      for (size_t s = 0; s < S; ++s) {
        for (size_t c = 0; c < C; ++c) {
          // input ((oh - r + pad) / stride, (ow - s + pad) / stride, c)
          //   where (oh - r + pad) % stride == 0 && (ow - s + pad) % stride == 0
          if ((oh - r + PAD) % STRIDE != 0 || (ow - s + PAD) % STRIDE != 0) continue;
          size_t ih = (oh - r + PAD) / STRIDE;
          size_t iw = (ow - s + PAD) / STRIDE;
          if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
          float ii = input[ih * W * C + iw * C + c];
          // filter (r, s, k, c)
          float ff = filter[r * S * K * C + s * K * C + k * C + c];
          x += ii * ff;
        }
      }
    }
    // output (oh, ow, k)
    if (ow < OW && oh < OH) output[oh * OW * K + ow * K + k] = x;
  }
}
