/*
 * Fast CUDA kernel for UTF-8 byte grouping using shared memory.
 * Supports seq_len up to 1024 (limited by shared memory).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_SEQ_LEN_SHARED 1024

__global__ void group_utf8_bytes_kernel_shared(
    const int64_t* __restrict__ input,
    int64_t* __restrict__ output,
    int* __restrict__ num_groups_out,
    const int batch_size,
    const int seq_len,
    const int max_groups,
    const int padding_value
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    extern __shared__ int shared_mem[];
    int* is_leading = shared_mem;
    int* group_idx = shared_mem + seq_len;
    int* group_start_pos = shared_mem + 2 * seq_len;
    int* group_lengths = shared_mem + 2 * seq_len + max_groups;

    const int64_t* input_row = input + batch_idx * seq_len;
    int64_t* output_row = output + batch_idx * max_groups * 4;

    // Initialize
    for (int i = threadIdx.x; i < max_groups; i += blockDim.x) {
        group_start_pos[i] = seq_len;
        group_lengths[i] = 0;
    }
    __syncthreads();

    // Phase 1: Identify leading bytes
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        int64_t byte_val = input_row[i];
        int non_padding = (byte_val != padding_value) ? 1 : 0;
        int leading = non_padding && (byte_val < 128 || byte_val >= 192);
        is_leading[i] = leading;
    }
    __syncthreads();

    // Sequential prefix sum (single thread)
    if (threadIdx.x == 0) {
        int cumsum = 0;
        for (int i = 0; i < seq_len; i++) {
            if (is_leading[i]) {
                group_start_pos[cumsum] = i;
                cumsum++;
            }
            group_idx[i] = (cumsum > 0) ? cumsum - 1 : 0;
        }
        num_groups_out[batch_idx] = cumsum;

        // Compute group lengths
        for (int g = 0; g < cumsum; g++) {
            int start = group_start_pos[g];
            int end = (g + 1 < cumsum) ? group_start_pos[g + 1] : seq_len;
            int len = 0;
            for (int j = start; j < end && j < seq_len; j++) {
                if (input_row[j] == padding_value) break;
                len++;
            }
            group_lengths[g] = (len > 4) ? 4 : len;
        }
    }
    __syncthreads();

    // Phase 2: Scatter bytes to output
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        int64_t byte_val = input_row[i];
        if (byte_val == padding_value) continue;

        int g = group_idx[i];
        int start = group_start_pos[g];
        int pos_in_group = i - start;
        if (pos_in_group >= 4) continue;

        int group_len = group_lengths[g];
        int out_pos = 4 - group_len + pos_in_group;
        if (out_pos >= 0 && out_pos < 4) {
            output_row[g * 4 + out_pos] = byte_val;
        }
    }
}


torch::Tensor group_utf8_bytes_cuda_shared(
    torch::Tensor input,
    int padding_value
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch, seq_len)");
    TORCH_CHECK(input.dtype() == torch::kInt64, "Input must be int64");

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);

    TORCH_CHECK(seq_len <= MAX_SEQ_LEN_SHARED,
                "Sequence length ", seq_len, " exceeds shared memory limit: ", MAX_SEQ_LEN_SHARED,
                ". Use group_utf8_bytes_cuda_global for larger sequences.");

    const int max_groups = seq_len;

    auto output = torch::zeros({batch_size, max_groups, 4}, input.options());
    auto num_groups = torch::zeros({batch_size}, input.options().dtype(torch::kInt32));

    const int shared_mem_size = (2 * seq_len + 2 * max_groups) * sizeof(int);
    const int threads_per_block = 256;

    group_utf8_bytes_kernel_shared<<<batch_size, threads_per_block, shared_mem_size>>>(
        input.data_ptr<int64_t>(),
        output.data_ptr<int64_t>(),
        num_groups.data_ptr<int>(),
        batch_size,
        seq_len,
        max_groups,
        padding_value
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    int actual_max_groups = num_groups.max().item<int>();
    if (actual_max_groups > 0 && actual_max_groups < max_groups) {
        output = output.slice(1, 0, actual_max_groups);
    }

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("group_utf8_bytes_cuda", &group_utf8_bytes_cuda_shared,
          "Fast UTF-8 grouping using shared memory (seq_len <= 1024)",
          py::arg("input"),
          py::arg("padding_value") = 0);
}
