#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define I2D(i,j,Nj) j + Nj*i


template <typename scalar_t>
__device__ scalar_t calculateDistance(
    size_t i_v, 
    size_t j_v, 
    const scalar_t *d_coord,
    size_t n_coords
    ){
    scalar_t distsq = 0;
    // if (i_v == j_v) return 0;
    for (size_t i = 0; i < n_coords; i++) {
        scalar_t dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist * dist;
    }
    return distsq;
    }

template <typename scalar_t>
__device__ scalar_t calculateDistanceDirectional(
        const size_t i_v,
        const size_t j_v,
        const scalar_t *d_coord_of,
        const scalar_t *d_coord_in,
        const size_t n_coords
){
    scalar_t distsq = 0;
    // if (i_v == j_v) return 0;
    for (size_t i = 0; i < n_coords; i++) {
        scalar_t dist = d_coord_of[I2D(i_v,i,n_coords)] - d_coord_in[I2D(j_v,i,n_coords)];
        distsq += dist * dist;
    }
    return distsq;
}


template <typename scalar_t>
__device__ int32_t searchLargestDistance(
    int32_t i_v, 
    scalar_t* d_dist, 
    int32_t n_neigh, 
    scalar_t& maxdist
    ){
    maxdist = 0;
    int32_t maxidx = 0;
    if (n_neigh < 2)
        return maxidx;
    for (int32_t n = 1; n < n_neigh; n++) { //0 is self
        scalar_t distsq = d_dist[I2D(i_v, n, n_neigh)];
        bool isgreater = distsq > maxdist;
        bool isless = !isgreater;
        maxdist = distsq*isgreater + maxdist*isless;
        maxidx = n*isgreater + maxidx*isless;
        }
    return maxidx;
}


template <typename scalar_t>
__device__ int32_t searchLargestDistanceDirectional(
        int32_t i_v,
        scalar_t* d_dist,
        int32_t n_neigh,
        scalar_t& maxdist
){
    maxdist = 0;
    int32_t maxidx = 0;
    if (n_neigh < 1)
        return maxidx;
    for (int32_t n = 0; n < n_neigh; n++) { //0 is self
        scalar_t distsq = d_dist[I2D(i_v, n, n_neigh)];
        bool isgreater = distsq > maxdist;
        bool isless = !isgreater;
        maxdist = distsq*isgreater + maxdist*isless;
        maxidx = n*isgreater + maxidx*isless;
    }
    return maxidx;
}



template <typename scalar_t> 
__global__ void set_defaults(
    scalar_t *d_dist,
    int32_t* d_indices,
    int32_t n_vert,
    int32_t n_neigh,
    bool directional)
{
    const int32_t i_v = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_v >= n_vert) return;

    const int32_t n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= n_neigh) return;

    // Initialize first neighbor as the self loop, other neighbors as -1
    if (n == 0 && !directional)
        d_indices[I2D(i_v, n, n_neigh)] = i_v;
    else
        d_indices[I2D(i_v, n, n_neigh)] = -1;

    d_dist[I2D(i_v, n, n_neigh)] = 0;
}

template <typename scalar_t> 
__global__ void select_knn_kernel(
    const scalar_t *d_coord,
    const int32_t *d_row_splits,
    const int32_t *d_mask,
    int32_t *d_indices,
    scalar_t *d_dist,

    const int32_t n_vert,
    const int32_t n_neigh,
    const int32_t n_coords,

    const int32_t j_rs,
    const scalar_t max_radius
    ){

    //really no buffering at all here

    const int32_t start_vert = d_row_splits[j_rs];
    const int32_t end_vert = d_row_splits[j_rs + 1];

    const int32_t i_v = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    if (i_v >= end_vert || i_v >= n_vert)
        return;//this will be a problem with actual RS

    //protection against n_vert<n_neigh
    int32_t max_neighbours = n_neigh;
    int32_t nvert_in_row = end_vert - start_vert;
    if (nvert_in_row < n_neigh) { max_neighbours = nvert_in_row; }

    int32_t nfilled = 1;
    int32_t maxidx_local = 0;
    scalar_t maxdistsq = 0;

    int32_t j_v = 0;
    for (j_v = start_vert; j_v < end_vert && nfilled < max_neighbours; j_v++) {
        scalar_t distsq = calculateDistance(i_v, j_v, d_coord, n_coords);
        if (i_v == j_v || (max_radius > 0 && distsq > max_radius)) continue;
        // if (max_radius > 0 && distsq > max_radius) continue;
        //fill up
        d_indices[I2D(i_v, nfilled, n_neigh)] = j_v;
        d_dist[I2D(i_v, nfilled, n_neigh)] = distsq;
        if (distsq > maxdistsq) {
            maxdistsq = distsq;
            maxidx_local = nfilled;
        }
        nfilled++;
    }
    // for the rest start from where we left off and only do index shuffling
    for(; j_v < end_vert; j_v++ ) {
        scalar_t distsq = calculateDistance(i_v, j_v, d_coord, n_coords);
        if (i_v == j_v || distsq > maxdistsq) continue;

        //replace former max
        d_indices[I2D(i_v, maxidx_local, n_neigh)] = j_v;
        d_dist[I2D(i_v, maxidx_local, n_neigh)] = distsq;
        //search new max
        maxidx_local = searchLargestDistance(i_v, d_dist, n_neigh, maxdistsq);
    }
}


std::tuple<torch::Tensor, torch::Tensor> select_knn_cuda_fn(
    torch::Tensor coords,
    torch::Tensor row_splits,
    torch::Tensor mask,
    int64_t n_neighbours,
    double max_radius,
    int64_t mask_mode)
{
    CHECK_CUDA(coords);
    CHECK_CUDA(row_splits);
    CHECK_CUDA(mask);

    const auto n_vert = coords.size(0);
    const auto n_coords = coords.size(1);
    const auto n_rs = row_splits.size(0);
    const auto n_neigh = n_neighbours;

    if (max_radius > 0) max_radius *= max_radius;

    auto output_dist_tensor = torch::zeros({ n_vert, n_neighbours },
        torch::TensorOptions().dtype(coords.dtype()).device(coords.device()));
    auto output_idx_tensor = torch::zeros({ n_vert, n_neighbours },
        torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));

    // get the grid and block values for parallel CUDA programming
    // Blocksize 256 over n_vert, blocksize 4 over n_neighbours
    dim3 block(256, 4);
    // Ensure enough blocks in the grid over these dims by rounding up
    dim3 grid((n_vert+block.x-1)/block.x, (n_neigh+block.y-1)/block.y);

    AT_DISPATCH_FLOATING_TYPES(coords.type(), "set_defaults", ([&] {
        set_defaults <scalar_t> <<<grid, block>>> (
            output_dist_tensor.data_ptr<scalar_t>(),
            output_idx_tensor.data_ptr<int32_t>(),
            n_vert,
            n_neigh,
            false);
    }));

    std::vector<int32_t> cpu_rowsplits(n_rs);
    cudaMemcpy(&cpu_rowsplits.at(0), row_splits.data_ptr<int32_t>(), n_rs * sizeof(int32_t), cudaMemcpyDeviceToHost);


    size_t block_size = 1024;
    size_t n_blocks;

    for (int32_t j_rs = 0; j_rs < n_rs - 1; j_rs++) {
        int32_t nvert_rs = cpu_rowsplits.at(j_rs + 1) - cpu_rowsplits.at(j_rs);
        // grid_and_block gb(nvert_rs, 1024);
        n_blocks = (nvert_rs + block_size - 1) / block_size;

        AT_DISPATCH_FLOATING_TYPES(coords.type(), "select_knn_kernel", ([&] {
            select_knn_kernel <scalar_t> <<<n_blocks, block_size>>> (
                    coords.data_ptr<scalar_t>(),
                    row_splits.data_ptr<int32_t>(),
                    mask.data_ptr<int32_t>(),
                    output_idx_tensor.data_ptr<int32_t>(),
                    output_dist_tensor.data_ptr<scalar_t>(),
                    
                    n_vert,
                    n_neigh,
                    n_coords,
                    
                    j_rs,
                    max_radius);
            }));
    }
    
    return std::make_tuple(output_idx_tensor, output_dist_tensor);

}


template <typename scalar_t>
__global__ void select_knn_directional_kernel(
        const scalar_t *d_coord_of,
        const int32_t *d_row_splits_of,
        const scalar_t *d_coord_in,
        const int32_t *d_row_splits_in,
        const int32_t *d_mask,
        int32_t *d_indices,
        scalar_t *d_dist,

        const int32_t n_vert,
        const int32_t n_neigh,
        const int32_t n_coords,

        const int32_t j_rs,
        const scalar_t max_radius
){

    //really no buffering at all here

    const int32_t start_vert_of = d_row_splits_of[j_rs];
    const int32_t end_vert_of = d_row_splits_of[j_rs + 1];

    const int32_t start_vert_in = d_row_splits_in[j_rs];
    const int32_t end_vert_in = d_row_splits_in[j_rs + 1];


    const int32_t i_v = blockIdx.x * blockDim.x + threadIdx.x + start_vert_of;
    if (i_v >= end_vert_of || i_v >= n_vert)
        return;//this will be a problem with actual RS

    //protection against n_vert<n_neigh
    int32_t max_neighbours = n_neigh;
    int32_t nvert_in_row = end_vert_in - start_vert_in;
    if (nvert_in_row < n_neigh) { max_neighbours = nvert_in_row; }

    int32_t nfilled = 0;
    int32_t maxidx_local = 0;
    scalar_t maxdistsq = 0;

    int32_t j_v = 0;
    for (j_v = start_vert_in; j_v < end_vert_in && nfilled < max_neighbours; j_v++) {
        scalar_t distsq = calculateDistanceDirectional(i_v, j_v, d_coord_of, d_coord_in, n_coords);
        if ((max_radius > 0 && distsq > max_radius)) continue;
        // if (max_radius > 0 && distsq > max_radius) continue;
        //fill up
        d_indices[I2D(i_v, nfilled, n_neigh)] = j_v;
        d_dist[I2D(i_v, nfilled, n_neigh)] = distsq;
        if (distsq > maxdistsq) {
            maxdistsq = distsq;
            maxidx_local = nfilled;
        }
        nfilled++;
    }
    // for the rest start from where we left off and only do index shuffling
    for(; j_v < end_vert_in; j_v++ ) {
        scalar_t distsq = calculateDistanceDirectional(i_v, j_v, d_coord_of, d_coord_in, n_coords);
        if (distsq > maxdistsq) continue;

        //replace former max
        d_indices[I2D(i_v, maxidx_local, n_neigh)] = j_v;
        d_dist[I2D(i_v, maxidx_local, n_neigh)] = distsq;
        //search new max
        maxidx_local = searchLargestDistanceDirectional(i_v, d_dist, n_neigh, maxdistsq);
    }
}



std::tuple<torch::Tensor, torch::Tensor> select_knn_directional_cuda_fn(
        torch::Tensor coords_of,
        torch::Tensor row_splits_of,
        torch::Tensor coords_in,
        torch::Tensor row_splits_in,
        torch::Tensor mask,
        int64_t n_neighbours,
        double max_radius,
        int64_t mask_mode)
{
    CHECK_CUDA(coords_of);
    CHECK_CUDA(row_splits_of);
    CHECK_CUDA(coords_in);
    CHECK_CUDA(row_splits_in);
    CHECK_CUDA(mask);

    const auto n_vert = coords_of.size(0);
    const auto n_coords = coords_of.size(1);
    const auto n_rs = row_splits_of.size(0);
    const auto n_neigh = n_neighbours;

    if (max_radius > 0) max_radius *= max_radius;

    auto output_dist_tensor = torch::zeros({ n_vert, n_neighbours },
                                           torch::TensorOptions().dtype(coords_of.dtype()).device(coords_of.device()));
    auto output_idx_tensor = torch::zeros({ n_vert, n_neighbours },
                                          torch::TensorOptions().dtype(torch::kInt32).device(coords_of.device()));

    // get the grid and block values for parallel CUDA programming
    // Blocksize 256 over n_vert, blocksize 4 over n_neighbours
    dim3 block(256, 4);
    // Ensure enough blocks in the grid over these dims by rounding up
    dim3 grid((n_vert+block.x-1)/block.x, (n_neigh+block.y-1)/block.y);

    AT_DISPATCH_FLOATING_TYPES(coords_of.type(), "set_defaults", ([&] {
        set_defaults <scalar_t> <<<grid, block>>> (
                output_dist_tensor.data_ptr<scalar_t>(),
                output_idx_tensor.data_ptr<int32_t>(),
                n_vert,
                n_neigh,
                true);
    }));

    std::vector<int32_t> cpu_rowsplits(n_rs);
    cudaMemcpy(&cpu_rowsplits.at(0), row_splits_of.data_ptr<int32_t>(), n_rs * sizeof(int32_t), cudaMemcpyDeviceToHost);


    size_t block_size = 1024;
    size_t n_blocks;

    for (int32_t j_rs = 0; j_rs < n_rs - 1; j_rs++) {
        int32_t nvert_rs = cpu_rowsplits.at(j_rs + 1) - cpu_rowsplits.at(j_rs);
        // grid_and_block gb(nvert_rs, 1024);
        n_blocks = (nvert_rs + block_size - 1) / block_size;

        AT_DISPATCH_FLOATING_TYPES(coords_of.type(), "select_knn_kernel", ([&] {
            select_knn_directional_kernel <scalar_t> <<<n_blocks, block_size>>> (
                    coords_of.data_ptr<scalar_t>(),
                    row_splits_of.data_ptr<int32_t>(),
                    coords_in.data_ptr<scalar_t>(),
                    row_splits_in.data_ptr<int32_t>(),
                    mask.data_ptr<int32_t>(),
                    output_idx_tensor.data_ptr<int32_t>(),
                    output_dist_tensor.data_ptr<scalar_t>(),

                    n_vert,
                    n_neigh,
                    n_coords,

                    j_rs,
                    max_radius);
        }));
    }

    return std::make_tuple(output_idx_tensor, output_dist_tensor);

}
