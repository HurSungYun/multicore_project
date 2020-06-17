#define BLOCK_SIZE 4
#define STRIDE 2
#define PAD 1
#define CACHE_SIZE 12

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

  __local float local_input[4 * 4 * CACHE_SIZE][BLOCK_SIZE * BLOCK_SIZE];
  __local float local_filter[CACHE_SIZE][4 * 4 * CACHE_SIZE];

  float x = k < K ? bias[k] : 0;

  int tiles = (C + CACHE_SIZE - 1) / CACHE_SIZE;

  int input_idx = local_oh * BLOCK_SIZE + local_ow;


  for (int t = 0; t < tiles; t++) {
    int c_base = t * CACHE_SIZE;
    int c_max = c_base + CACHE_SIZE > C ? C - c_base : CACHE_SIZE;

    for (int r = 0; r < R; r++) {
      for (int s = 0; s < S; s++) {
        for (int c_offset = 0; c_offset < c_max; c_offset++) {
          int idx = r * 4 * CACHE_SIZE + s * CACHE_SIZE + c_offset;
          if (ow < OW && oh < OH) {
            local_input[idx][input_idx] = input[oh * OW * R * S * C + ow * R * S * C + r * S * C + s * C + (c_base + c_offset)]; // input[idx * OH * OW + oh * OW + ow];
          } else {
            local_input[idx][input_idx] = 0;
          }

          if (k < K) {
            local_filter[local_k][idx] = filter[r * S * C * K + s * C * K + (c_base + c_offset) * K + k];
          } else {
            local_filter[local_k][idx] = 0;
          }
        }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
 
    for (int r = 0; r < R; r++) {
      for (int s = 0; s < S; s++) {
        for (int c_offset = 0; c_offset < c_max; c_offset++) {
          int idx = r * 4 * CACHE_SIZE + s * CACHE_SIZE + c_offset;
          x += local_input[idx][input_idx] * local_filter[local_k][idx];
        }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (ow < OW && oh < OH && k < K) output[oh * OW * K + ow * K + k] = x;
  /*
  __local float local_input[BLOCK_SIZE][BLOCK_SIZE][CACHE_SIZE];

  float x = k < K ? bias[k] : 0;

  int tiles = (C + CACHE_SIZE - 1) / CACHE_SIZE;
  for (int r = 0; r < R; r++) {
    int ih = oh * STRIDE - PAD + r;
    if (ih < 0 || ih >= H) continue;
    for (int s = 0; s < S; s++) {
      int iw = ow * STRIDE - PAD + s;
      if (iw < 0 || iw >= W) continue;
      for (int t = 0; t < tiles; t++) {
        int c_base = t * CACHE_SIZE;
        int c = c_base + local_c;
        if (c < C) {
          local_input[local_oh][local_ow][local_c] = input[ih * W * C + iw * C + c_base + local_c];
        } else {
          local_input[local_oh][local_ow][local_c] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (k < K) {
          int c_max = c_base + CACHE_SIZE > C ? C - c_base : CACHE_SIZE;

          for (int c_offset = 0; c_offset < c_max; c_offset++) {
            x = fma(local_input[local_oh][local_ow][c_offset], filter[r * S * C * K + s * C * K + (c_base + c_offset) * K + k], x);
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
  }
  if (ow < OW && oh < OH && k < K) output[oh * OW * K + ow * K + k] = x;
  */
}


__kernel void conv2d_transpose(__global float *input, __global float *filter, __global float *bias, __global float *output, int H, int W, int C, int R, int S, int K) {
  int local_oh = get_local_id(0);
  int local_ow = get_local_id(1);
  int local_k = get_local_id(2);
  int oh = get_group_id(0) * BLOCK_SIZE + local_oh;
  int ow = get_group_id(1) * BLOCK_SIZE + local_ow;
  int k = get_group_id(2) * CACHE_SIZE + local_k;

  int local_c = local_k;

  int OH = H * STRIDE;
  int OW = W * STRIDE;

  __local float local_input[BLOCK_SIZE][BLOCK_SIZE][CACHE_SIZE];
//  __local float local_filter[CACHE_SIZE][CACHE_SIZE]; // [k][c]

  float x = k < K ? bias[k] : 0;

  int tiles = (C + CACHE_SIZE - 1) / CACHE_SIZE;
  for (int r = 0; r < R; ++r) {
    int ih = (oh - r + PAD) / STRIDE;
    for (int s = 0; s < S; ++s) {
      int iw = (ow - s + PAD) / STRIDE;
      
      if ((oh - r + PAD) % STRIDE != 0 || ih < 0 || ih >= H || (ow - s + PAD) % STRIDE != 0 || iw < 0 || iw >= W) {
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_LOCAL_MEM_FENCE);
      } else {

      for (int t = 0; t < tiles; t++) {
        int c_base = t * CACHE_SIZE;
        int c = c_base + local_c;
        int c_max = c_base + CACHE_SIZE > C ? C - c_base : CACHE_SIZE;

        if (c < C) {
          local_input[local_oh][local_ow][local_c] = input[ih * W * C + iw * C + c_base + local_c];
        } else {
          local_input[local_oh][local_ow][local_c] = 0;
        }
/*
        if (k < K) {
          for (int c_offset = 0; c_offset < c_max; c_offset++) {
            local_filter[local_k][c_offset] = filter[r * S * K * C + s * K * C + k * C + (c_base + c_offset)];
          }
        } else {
          for (int c_offset = 0; c_offset < c_max; c_offset++) {
            local_filter[local_k][c_offset] = 0;
          } 
        }
*/
        barrier(CLK_LOCAL_MEM_FENCE);

        if (k < K) {

          for (int c_offset = 0; c_offset < c_max; c_offset++) {
//            x = fma(local_input[c_offset], local_filter[local_k][c_offset], x);
            x = fma(local_input[local_oh][local_ow][c_offset], filter[r * S * K * C + s * K * C + k * C + (c_base + c_offset)], x);
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
