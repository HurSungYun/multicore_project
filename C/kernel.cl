#define BLOCK_SIZE 4
#define BLOCK_SIZE_TRANS 4
#define STRIDE 2
#define PAD 1
#define CACHE_SIZE 31

__kernel void conv2d(__global float *input, __global float *filter, __global float *bias, __global float *output, int H, int W, int C, int R, int S, int K) {
  int local_oh = get_local_id(0);
  int local_ow = get_local_id(1);
  int local_k = get_local_id(2);
  int oh = get_group_id(0) * BLOCK_SIZE + local_oh;
  int ow = get_group_id(1) * BLOCK_SIZE + local_ow;
  int k = get_group_id(2) * CACHE_SIZE + local_k;
  
  int local_c = local_k;

  int OH = H / STRIDE;
  int OW = W / STRIDE;

  __local float local_input[2 * BLOCK_SIZE + 2][2 * BLOCK_SIZE + 2][CACHE_SIZE];

  float x = k < K ? bias[k] : 0;

  int tiles = (C + CACHE_SIZE - 1) / CACHE_SIZE;

  for (int t = 0; t < tiles; t++) {
    int c_base = t * CACHE_SIZE;
    int c = c_base + local_c;

    for (int r = 0; r < R; r++) {
      for (int s = 0; s < S; s++) {
        int ih = oh * STRIDE - PAD + r;
        int iw = ow * STRIDE - PAD + s;

        if (ih < 0 || ih >= H || iw < 0 || iw >= W || c >= C) {
          local_input[local_oh * STRIDE + r][local_ow * STRIDE + s][local_c] = 0;
        } else {
          local_input[local_oh * STRIDE + r][local_ow * STRIDE + s][local_c] = input[ih * W * C + iw * C + c_base + local_c];
        }
      }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (k < K) {
      int c_max = c_base + CACHE_SIZE > C ? C - c_base : CACHE_SIZE;

//      for (int r = 0; r < R; r++) {
//        for (int s = 0; s < S; s++) {
          for (int c_offset = 0; c_offset < c_max; c_offset++) {
            x = fma(local_input[local_oh * STRIDE + 0][local_ow * STRIDE + 0][c_offset], filter[(c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 0][local_ow * STRIDE + 1][c_offset], filter[C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 0][local_ow * STRIDE + 2][c_offset], filter[2 * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 0][local_ow * STRIDE + 3][c_offset], filter[3 * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 1][local_ow * STRIDE + 0][c_offset], filter[1 * S * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 1][local_ow * STRIDE + 1][c_offset], filter[1 * S * C * K + C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 1][local_ow * STRIDE + 2][c_offset], filter[1 * S * C * K + 2 * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 1][local_ow * STRIDE + 3][c_offset], filter[1 * S * C * K + 3 * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 2][local_ow * STRIDE + 0][c_offset], filter[2 * S * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 2][local_ow * STRIDE + 1][c_offset], filter[2 * S * C * K + C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 2][local_ow * STRIDE + 2][c_offset], filter[2 * S * C * K + 2 * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 2][local_ow * STRIDE + 3][c_offset], filter[2 * S * C * K + 3 * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 3][local_ow * STRIDE + 0][c_offset], filter[3 * S * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 3][local_ow * STRIDE + 1][c_offset], filter[3 * S * C * K + C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 3][local_ow * STRIDE + 2][c_offset], filter[3 * S * C * K + 2 * C * K + (c_base + c_offset) * K + k], x);
            x = fma(local_input[local_oh * STRIDE + 3][local_ow * STRIDE + 3][c_offset], filter[3 * S * C * K + 3 * C * K + (c_base + c_offset) * K + k], x);
          }
//        }
//      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (ow < OW && oh < OH && k < K) output[oh * OW * K + ow * K + k] = x;
}


__kernel void conv2d_transpose(__global float *input, __global float *filter, __global float *bias, __global float *output, int H, int W, int C, int R, int S, int K) {
  int local_oh = get_local_id(0);
  int local_ow = get_local_id(1);
  int local_k = get_local_id(2);
  int oh = get_group_id(0) * BLOCK_SIZE_TRANS + local_oh;
  int ow = get_group_id(1) * BLOCK_SIZE_TRANS + local_ow;
  int k = get_group_id(2) * CACHE_SIZE + local_k;

  int local_c = local_k;

  int OH = H * STRIDE;
  int OW = W * STRIDE;

  __local float local_input[BLOCK_SIZE_TRANS][BLOCK_SIZE_TRANS][CACHE_SIZE];

  float x = k < K ? bias[k] : 0;

  int tiles = (C + CACHE_SIZE - 1) / CACHE_SIZE;

  


  // (0, 0)

  for (int r = (oh - 0 + PAD) % STRIDE == 0 ? 0 : 1; r < R; r += STRIDE) {
    int ih = (oh - r + PAD) / STRIDE;
    for (int s = (ow - 0 + PAD) % STRIDE == 0 ? 0 : 1; s < S; s += STRIDE) {
      int iw = (ow - s + PAD) / STRIDE;

      if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
      for (int t = 0; t < tiles; t++) {
        int c_base = t * CACHE_SIZE;
        int c = c_base + local_c;
        int c_max = c_base + CACHE_SIZE > C ? C - c_base : CACHE_SIZE;

       if (c < C) {
          local_input[local_oh][local_ow][local_c] = input[ih * W * C + iw * C + c_base + local_c];
        } else {
          local_input[local_oh][local_ow][local_c] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (k < K) {

          for (int c_offset = 0; c_offset < c_max; c_offset++) {
            x = fma(local_input[local_oh][local_ow][c_offset], filter[r * S * K * C + s * K * C + k * C + (c_base + c_offset)], x);
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
  }


  // output (oh, ow, k)
  if (ow < OW && oh < OH && k < K) output[oh * OW * K + ow * K + k] = x;
}
