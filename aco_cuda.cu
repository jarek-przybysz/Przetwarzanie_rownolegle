//Ant Colony Optimization for TSP – CUDA version
// !nvcc -o aco_runner aco_cuda.cu -arch=sm_75 -O3
// ./aco_runner berlin52.tsp eil101.tsp eil51.tsp pr107.tsp rat99.tsp st70.tsp u159.tsp
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <filesystem>
#include <cfloat>

#include <cuda_runtime.h>
#include <curand_kernel.h>

struct City
{
    int id;
    float x, y;
};

int num_cities = 0;
std::vector<City> cities_nodes;
std::string tsp_name;

int num_ants = 300;
float alpha = 1.0f;
float beta = 3.0f;
float evaporation_rate = 0.25f;
float Q = 100.0f;
int max_iterations = 750;
float initial_pheromone = 0.1f;

__device__ __constant__ int d_c_num_cities;
__device__ __constant__ int d_c_num_ants;
__device__ __constant__ float d_c_alpha;
__device__ __constant__ float d_c_beta;
__device__ __constant__ float d_c_evaporation_rate;
__device__ __constant__ float d_c_Q;

#define CHECK_CUDA(call)                                                                                       \
    do                                                                                                         \
    {                                                                                                          \
        cudaError_t err = call;                                                                                \
        if (err != cudaSuccess)                                                                                \
        {                                                                                                      \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                \
        }                                                                                                      \
    } while (0)

bool parse_tsp_file(const std::string &filename, float *&h_distance_matrix, float *&h_heuristic_matrix);
void run_aco_on_gpu(float *h_distance_matrix, float *h_heuristic_matrix, std::vector<int> &best_tour_overall, float &best_length_overall);
inline void eat_ws_and_colon(std::stringstream &ss);

float calculate_euc_2d_distance(const City &a, const City &b) { return roundf(sqrtf(powf(a.x - b.x, 2) + powf(a.y - b.y, 2))); }
float calculate_att_distance(const City &a, const City &b) { return roundf(sqrtf((powf(a.x - b.x, 2) + powf(a.y - b.y, 2)) / 10.0f)); }
float calculate_ceil_2d_distance(const City &a, const City &b) { return ceilf(sqrtf(powf(a.x - b.x, 2) + powf(a.y - b.y, 2))); }

bool parse_tsp_file(const std::string &filename, float *&h_distance_matrix, float *&h_heuristic_matrix)
{
    std::ifstream in(filename);
    if (!in)
    {
        std::cerr << "Nie mogę otworzyć " << filename << "\n";
        return false;
    }
    std::string line, key, edge_type = "";
    bool reading_coords = false;
    cities_nodes.clear();
    num_cities = 0;
    while (std::getline(in, line))
    {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        if (line.empty())
            continue;
        std::stringstream ss(line);
        if (!reading_coords)
        {
            ss >> key;
            std::transform(key.begin(), key.end(), key.begin(), ::toupper);
            if (key == "NAME" || key == "NAME:")
            {
                eat_ws_and_colon(ss);
                ss >> tsp_name;
            }
            else if (key == "DIMENSION" || key == "DIMENSION:")
            {
                eat_ws_and_colon(ss);
                ss >> num_cities;
            }
            else if (key == "EDGE_WEIGHT_TYPE" || key == "EDGE_WEIGHT_TYPE:")
            {
                eat_ws_and_colon(ss);
                ss >> edge_type;
                std::transform(edge_type.begin(), edge_type.end(), edge_type.begin(), ::toupper);
            }
            else if (key == "NODE_COORD_SECTION")
            {
                if (num_cities == 0)
                    return false;
                cities_nodes.assign(num_cities, {});
                reading_coords = true;
                for (int i = 0; i < num_cities; ++i)
                {
                    int id;
                    float x, y;
                    if (!(in >> id >> x >> y))
                        return false;
                    cities_nodes[id - 1] = {id, x, y};
                }
                reading_coords = false;
            }
            else if (key == "EOF")
            {
                break;
            }
        }
    }
    if (cities_nodes.size() != static_cast<size_t>(num_cities))
        return false;
    h_distance_matrix = new float[num_cities * num_cities];
    h_heuristic_matrix = new float[num_cities * num_cities];
    for (int i = 0; i < num_cities; ++i)
    {
        for (int j = 0; j < num_cities; ++j)
        {
            float d = 0.0f;
            if (i != j)
            {
                if (edge_type == "EUC_2D" || edge_type == ":" || edge_type.empty())
                    d = calculate_euc_2d_distance(cities_nodes[i], cities_nodes[j]);
                else
                {
                    std::cerr << "EDGE_WEIGHT_TYPE " << edge_type << " nieobsługiwany\n";
                    return false;
                }
            }
            h_distance_matrix[i * num_cities + j] = d;
            h_heuristic_matrix[i * num_cities + j] = (i != j && d != 0) ? 1.0f / d : 0.0f;
        }
    }
    return true;
}
inline void eat_ws_and_colon(std::stringstream &ss)
{
    ss >> std::ws;
    if (ss.peek() == ':')
        ss.get();
    ss >> std::ws;
}

__global__ void setup_curand_kernel(curandState *states, unsigned long seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < d_c_num_ants)
    {
        curand_init(seed, id, 0, &states[id]);
    }
}

__global__ void construct_tours_kernel(float *d_distance_matrix, float *d_pheromone_matrix, float *d_heuristic_matrix,
                                       int *d_tours, float *d_tour_lengths, bool *d_visited, curandState *d_rand_states)
{
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant_id >= d_c_num_ants)
        return;

    curandState local_rand_state = d_rand_states[ant_id];
    int *my_tour = d_tours + ant_id * d_c_num_cities;
    bool *my_visited = d_visited + ant_id * d_c_num_cities;

    for (int i = 0; i < d_c_num_cities; ++i)
        my_visited[i] = false;

    int start_node = curand(&local_rand_state) % d_c_num_cities;
    my_tour[0] = start_node;
    my_visited[start_node] = true;
    int current_city = start_node;
    float current_tour_length = 0.0f;

    for (int step = 1; step < d_c_num_cities; ++step)
    {
        float probabilities_sum = 0.0f;
        for (int next_city_candidate = 0; next_city_candidate < d_c_num_cities; ++next_city_candidate)
        {
            if (!my_visited[next_city_candidate])
            {
                float pheromone = d_pheromone_matrix[current_city * d_c_num_cities + next_city_candidate];
                float heuristic = d_heuristic_matrix[current_city * d_c_num_cities + next_city_candidate];
                probabilities_sum += powf(pheromone, d_c_alpha) * powf(heuristic, d_c_beta);
            }
        }

        float rand_val = curand_uniform(&local_rand_state) * probabilities_sum;
        float current_sum = 0.0f;
        int next_city = -1;
        for (int city_idx = 0; city_idx < d_c_num_cities; ++city_idx)
        {
            if (!my_visited[city_idx])
            {
                float pheromone = d_pheromone_matrix[current_city * d_c_num_cities + city_idx];
                float heuristic = d_heuristic_matrix[current_city * d_c_num_cities + city_idx];
                current_sum += powf(pheromone, d_c_alpha) * powf(heuristic, d_c_beta);
                if (current_sum >= rand_val)
                {
                    next_city = city_idx;
                    break;
                }
            }
        }
        if (next_city == -1)
        {
            for (int city_idx = 0; city_idx < d_c_num_cities; ++city_idx)
            {
                if (!my_visited[city_idx])
                {
                    next_city = city_idx;
                    break;
                }
            }
        }

        my_tour[step] = next_city;
        my_visited[next_city] = true;
        current_tour_length += d_distance_matrix[current_city * d_c_num_cities + next_city];
        current_city = next_city;
    }

    current_tour_length += d_distance_matrix[current_city * d_c_num_cities + start_node];
    d_tour_lengths[ant_id] = current_tour_length;
    d_rand_states[ant_id] = local_rand_state;
}

__global__ void find_best_ant_kernel(const float *d_tour_lengths, float *d_best_len_iter, int *d_best_idx_iter)
{
    extern __shared__ char shared_mem[];
    float *s_lengths = (float *)shared_mem;
    int *s_indices = (int *)&s_lengths[blockDim.x];

    int tid = threadIdx.x;
    if (tid < d_c_num_ants)
    {
        s_lengths[tid] = d_tour_lengths[tid];
        s_indices[tid] = tid;
    }
    else
    {
        s_lengths[tid] = FLT_MAX;
        s_indices[tid] = -1;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (s_lengths[tid] > s_lengths[tid + s])
            {
                s_lengths[tid] = s_lengths[tid + s];
                s_indices[tid] = s_indices[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        *d_best_len_iter = s_lengths[0];
        *d_best_idx_iter = s_indices[0];
    }
}

__global__ void update_global_best_kernel(float *d_best_len_iter, int *d_best_idx_iter,
                                          float *d_global_best_len, int *d_global_best_tour,
                                          const int *d_tours)
{
    if (*d_best_len_iter < *d_global_best_len)
    {
        *d_global_best_len = *d_best_len_iter;
        int best_ant_offset = (*d_best_idx_iter) * d_c_num_cities;
        const int *best_tour_src = d_tours + best_ant_offset;
        for (int i = 0; i < d_c_num_cities; ++i)
        {
            d_global_best_tour[i] = best_tour_src[i];
        }
    }
}

__global__ void evaporation_kernel(float *d_pheromone_matrix)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_dim = d_c_num_cities * d_c_num_cities;
    if (idx < matrix_dim)
    {
        d_pheromone_matrix[idx] *= (1.0f - d_c_evaporation_rate);
    }
}

__global__ void deposit_kernel(float *d_pheromone_matrix, const int *d_tours,
                               const float *d_best_len_iter, const int *d_best_idx_iter)
{
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx < d_c_num_cities)
    {
        float best_len = *d_best_len_iter;
        if (best_len < 1e-10f)
            best_len = 1.0f;

        int best_idx = *d_best_idx_iter;
        const int *d_best_tour_iter = d_tours + best_idx * d_c_num_cities;

        int u = d_best_tour_iter[edge_idx];
        int v = d_best_tour_iter[(edge_idx + 1) % d_c_num_cities];
        float deposit_amount = d_c_Q / best_len;

        atomicAdd(&d_pheromone_matrix[u * d_c_num_cities + v], deposit_amount);
        atomicAdd(&d_pheromone_matrix[v * d_c_num_cities + u], deposit_amount);
    }
}

void run_aco_on_gpu(float *h_distance_matrix, float *h_heuristic_matrix, std::vector<int> &best_tour_overall, float &best_length_overall)
{
    float *d_distance_matrix, *d_pheromone_matrix, *d_heuristic_matrix, *d_tour_lengths;
    int *d_tours, *d_global_best_tour, *d_best_idx_iter;
    bool *d_visited;
    curandState *d_rand_states;
    float *d_global_best_len, *d_best_len_iter;

    size_t matrix_size = (size_t)num_cities * num_cities * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_distance_matrix, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_pheromone_matrix, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_heuristic_matrix, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_tours, (size_t)num_ants * num_cities * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tour_lengths, num_ants * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_visited, (size_t)num_ants * num_cities * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_rand_states, num_ants * sizeof(curandState)));
    CHECK_CUDA(cudaMalloc(&d_global_best_tour, num_cities * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_global_best_len, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_best_len_iter, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_best_idx_iter, sizeof(int)));

    std::vector<float> h_pheromone_matrix(num_cities * num_cities, initial_pheromone);
    float h_global_best_len = FLT_MAX;
    CHECK_CUDA(cudaMemcpy(d_distance_matrix, h_distance_matrix, matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_heuristic_matrix, h_heuristic_matrix, matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pheromone_matrix, h_pheromone_matrix.data(), matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_global_best_len, &h_global_best_len, sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpyToSymbol(d_c_num_cities, &num_cities, sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_c_num_ants, &num_ants, sizeof(int)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_c_alpha, &alpha, sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_c_beta, &beta, sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_c_evaporation_rate, &evaporation_rate, sizeof(float)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_c_Q, &Q, sizeof(float)));

    int threads_per_block = 256;
    int tour_blocks = (num_ants + threads_per_block - 1) / threads_per_block;
    setup_curand_kernel<<<tour_blocks, threads_per_block>>>(d_rand_states, time(0));
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int it = 0; it < max_iterations; ++it)
    {
        construct_tours_kernel<<<tour_blocks, threads_per_block>>>(d_distance_matrix, d_pheromone_matrix, d_heuristic_matrix, d_tours, d_tour_lengths, d_visited, d_rand_states);
        CHECK_CUDA(cudaDeviceSynchronize());

        size_t shared_mem_size = (sizeof(float) + sizeof(int)) * 1024;
        find_best_ant_kernel<<<1, 1024, shared_mem_size>>>(d_tour_lengths, d_best_len_iter, d_best_idx_iter);
        CHECK_CUDA(cudaDeviceSynchronize());

        update_global_best_kernel<<<1, 1>>>(d_best_len_iter, d_best_idx_iter, d_global_best_len, d_global_best_tour, d_tours);
        CHECK_CUDA(cudaDeviceSynchronize());

        int evap_blocks = (num_cities * num_cities + threads_per_block - 1) / threads_per_block;
        evaporation_kernel<<<evap_blocks, threads_per_block>>>(d_pheromone_matrix);
        CHECK_CUDA(cudaDeviceSynchronize());

        int deposit_blocks = (num_cities + threads_per_block - 1) / threads_per_block;
        deposit_kernel<<<deposit_blocks, threads_per_block>>>(d_pheromone_matrix, d_tours, d_best_len_iter, d_best_idx_iter);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(&best_length_overall, d_global_best_len, sizeof(float), cudaMemcpyDeviceToHost));
    best_tour_overall.resize(num_cities);
    CHECK_CUDA(cudaMemcpy(best_tour_overall.data(), d_global_best_tour, num_cities * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_distance_matrix));
    CHECK_CUDA(cudaFree(d_pheromone_matrix));
    CHECK_CUDA(cudaFree(d_heuristic_matrix));
    CHECK_CUDA(cudaFree(d_tours));
    CHECK_CUDA(cudaFree(d_tour_lengths));
    CHECK_CUDA(cudaFree(d_visited));
    CHECK_CUDA(cudaFree(d_rand_states));
    CHECK_CUDA(cudaFree(d_global_best_tour));
    CHECK_CUDA(cudaFree(d_global_best_len));
    CHECK_CUDA(cudaFree(d_best_len_iter));
    CHECK_CUDA(cudaFree(d_best_idx_iter));
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Błąd: Podaj przynajmniej jeden plik .tsp jako argument." << std::endl;
        std::cerr << "Użycie: " << argv[0] << " <plik1.tsp> [plik2.tsp] ..." << std::endl;
        return 1;
    }

    std::ofstream log("results_tsp_cuda.txt", std::ios::app);
    log << "NAME\tBEST LEN\tTIME (ms)" << std::endl;

    for (int i = 1; i < argc; ++i)
    {
        const std::string filename = argv[i];
        std::cout << "\n[" << i << "/" << argc - 1 << "] Przetwarzanie pliku: " << filename << "\n";

        float *h_distance_matrix = nullptr, *h_heuristic_matrix = nullptr;
        if (!parse_tsp_file(filename, h_distance_matrix, h_heuristic_matrix))
        {
            std::cerr << "Błąd parsowania pliku " << filename << std::endl;
            delete[] h_distance_matrix;
            delete[] h_heuristic_matrix;
            continue;
        }

        std::vector<int> best_tour;
        float best_len;

        auto t0 = std::chrono::high_resolution_clock::now();
        run_aco_on_gpu(h_distance_matrix, h_heuristic_matrix, best_tour, best_len);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "Najlepsza znaleziona długość: " << std::fixed << std::setprecision(2) << best_len
                  << "        czas: " << ms << " ms\n";

        if (log)
        {
            log << tsp_name << '\t' << std::fixed << std::setprecision(2) << best_len << '\t' << ms << std::endl;
        }

        delete[] h_distance_matrix;
        delete[] h_heuristic_matrix;
    }

    return 0;
}
