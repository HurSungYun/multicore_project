#define BLOCK_SIZE 4
#define STRIDE 2
#define PAD 1
#define FILTER_SIZE 4

__kernel void conv2d(__global float *input, __global float *filter, __global float *bias, __global float *output, int H, int W, int C, int R, int S, int K) {
  int local_h = get_local_id(0);
  int local_w = get_local_id(1);
  int oh = get_group_id(0) * BLOCK_SIZE + get_local_id(0);
  int ow = get_group_id(1) * BLOCK_SIZE + get_local_id(1);

  int OH = H / STRIDE;
  int OW = W / STRIDE;
/*
  __local float local_input[BLOCK_SIZE * FILTER_SIZE + 2 * PAD][BLOCK_SIZE * FILTER_SIZE + 2 * PAD][BLOCK_SIZE];

  for (int k = 0; k < K; k++) {
    float x = bias[k];
    for (int cc = 0; cc < C; cc += BLOCK_SIZE) {

      int c_end = cc + BLOCK_SIZE > C ? C : cc + BLOCK_SIZE;
      for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
          for (int c = cc; c < c_end; c++) {
            int ih = oh * STRIDE - PAD + i;
            int iw = ow * STRIDE - PAD + j;
            if (ih < 0 || ih >= H || iw < 0 || iw >= W)
              local_input[local_h * FILTER_SIZE + i][local_w * FILTER_SIZE + j][c - cc] = 0.0f;
            else
              local_input[local_h * FILTER_SIZE + i][local_w * FILTER_SIZE + j][c - cc] = input[ih * W * C + iw * C + c];
          }
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int r = 0; r < FILTER_SIZE; r++) {
        for (int s = 0; s < FILTER_SIZE; s++) {      
          for (int c = cc; c < c_end; c++) {
            int ih = local_h * STRIDE + r; // - PAD + r;
            int iw = local_w * STRIDE + s; // - PAD + s;

//            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
            if (ih < 0 || iw < 0) continue;

            x += local_input[ih][iw][c - cc] * filter[r * S * C * K + s * C * K + c * K + k];
          }
        }
      }

      barrier(CLK_LOCAL_MEM_FENCE);

    }

    // K -> R -> S -> C

    output[oh * OW * K + ow * K + k] = x;
  }
*/

  for (int k = 0; k < K; k++) {
    float x = bias[k];
    for (int c = 0; c < C; ++c) {
      for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
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
  }
}
