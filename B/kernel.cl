#define BLOCK_SIZE 2
#define STRIDE 2
#define PAD 1
#define CACHE_SIZE 64

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

  int tiles = (C + CACHE_SIZE - 1) / CACHE_SIZE;
  for (int r = 0; r < R; r++) {
    for (int s = 0; s < S; s++) {
      int iw = ow * STRIDE - PAD + s;
      int ih = oh * STRIDE - PAD + r;
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

}


__kernel void conv2d_transpose(__global float *input, __global float *filter, __global float *bias, __global float *output, int H, int W, int C, int R, int S, int K) {
  int local_oh = get_local_id(0);
  int local_ow = get_local_id(1);
  int local_k = get_local_id(2);
  int oh = get_global_id(0);
  int ow = get_global_id(1);
  int k = get_group_id(2) * CACHE_SIZE + local_k;

  int local_c = local_k;

  int OH = H * STRIDE;
  int OW = W * STRIDE;

  __local float local_input[CACHE_SIZE];

  float x = k < K ? bias[k] : 0;

  int tiles = (C + CACHE_SIZE - 1) / CACHE_SIZE;
  for (int r = 0; r < R; ++r) {
    for (int s = 0; s < S; ++s) {
//      if ((oh - r + PAD) % STRIDE != 0 || (ow - s + PAD) % STRIDE != 0) continue;
      int ih = (oh - r + PAD) / STRIDE;
      int iw = (ow - s + PAD) / STRIDE;

      for (int t = 0; t < tiles; t++) {
        int c_base = t * CACHE_SIZE;
        int c = c_base + local_c;

        if ((oh - r + PAD) % STRIDE != 0 || (ow - s + PAD) % STRIDE != 0 || ih < 0 || ih >= H || iw < 0 || iw >= W) {
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
              x += local_input[c_offset] * filter[r * S * K * C + s * K * C + k * C + (c_base + c_offset)];
            }
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
      }
    }
  }
  // output (oh, ow, k)
  if (ow < OW && oh < OH && k < K) output[oh * OW * K + ow * K + k] = x;
}
