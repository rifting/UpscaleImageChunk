extern "C" __global__ void ssd_kernel(
    unsigned char* full, unsigned char* chunk,
    int full_w, int full_h, int chunk_w, int chunk_h,
    double* scores
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_x = full_w - chunk_w + 1;
    int max_y = full_h - chunk_h + 1;

    if (idx >= max_x * max_y) return;

    int x = idx % max_x;
    int y = idx / max_x;

    unsigned long long sum = 0;

    for (int j = 0; j < chunk_h; j++) {
        for (int i = 0; i < chunk_w; i++) {
            int full_idx = ((y + j) * full_w + (x + i)) * 4;
            int chunk_idx = (j * chunk_w + i) * 4;

            int dr = full[full_idx + 0] - chunk[chunk_idx + 0];
            int dg = full[full_idx + 1] - chunk[chunk_idx + 1];
            int db = full[full_idx + 2] - chunk[chunk_idx + 2];

            sum += dr * dr + dg * dg + db * db;
        }
    }

    // Normalize by number of pixels, otherwise the smaller images will always "win"
    scores[idx] = (double)sum / (chunk_w * chunk_h);
}
