// Ant Colony Optimization for TSP – OpenMP-parallel version
// g++-14 aco_par.cpp -o aco_par -std=c++17 -O2 -fopenmp
// ./aco_par berlin52.tsp eil101.tsp eil51.tsp pr107.tsp rat99.tsp st70.tsp u159.tsp

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <limits>
#include <omp.h>      
#include <filesystem> 

// -----------------------------  STRUKTURY DANYCH  -----------------------------
struct City
{
    int id;
    double x, y;
};

// ---------------------------  GLOBALNE PARAMETRY  -----------------------------
int num_cities = 0;
std::vector<City> cities_nodes;
std::vector<std::vector<double>> distance_matrix;
std::vector<std::vector<double>> heuristic_matrix;
std::vector<std::vector<double>> pheromone_matrix;
std::string tsp_name;
namespace fs = std::filesystem;
// ACO parameters (can be tuned)
int num_ants = 300;
double alpha = 1.0;
double beta = 3.0;
double evaporation_rate = 0.25;
double Q = 100.0;
int max_iterations = 750;
double initial_pheromone = 0.1;

// ---------------------------  ANT STRUCTURE  ----------------------------------
struct Ant
{
    std::vector<int> tour;
    std::vector<bool> visited;
    double tour_length = 0.0;

    void initialize(int n)
    {
        tour.clear();
        tour.reserve(n);
        visited.assign(n, false);
        tour_length = 0.0;
    }
    void reset()
    {
        tour.clear();
        std::fill(visited.begin(), visited.end(), false);
        tour_length = 0.0;
    }
};

// ---------------------------  RANDOMNESS UTIL  --------------------------------
// Each thread will have its own RNG.
static thread_local std::mt19937 tls_rng;
inline void init_thread_rng()
{
    static std::random_device rd;
    // Combine random_device with time and thread num for uniqueness
    unsigned seed = rd() ^ (static_cast<unsigned>(std::chrono::high_resolution_clock::now().time_since_epoch().count())) ^ omp_get_thread_num();
    tls_rng.seed(seed);
}

// ---------------------------  DEKLARACJE FUNKCJI  -----------------------------
bool parse_tsp_file(const std::string &file);
void initialize_aco_structs();
double calc_euc2d(const City &a, const City &b);
int select_next_city(int current, const Ant &ant);
void construct_tour(Ant &ant);
void update_pheromones(const std::vector<int> &best_tour, double best_len);
void run_aco(std::vector<int> &best_tour, double &best_len);

// ----------------------  IMPLEMENTACJA FUNKCJI POMOCNICZYCH  ------------------
double calc_euc2d(const City &a, const City &b)
{
    return std::round(std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)));
}
inline void eat_ws_and_colon(std::stringstream &ss)
{
    ss >> std::ws;        
    if (ss.peek() == ':') 
        ss.get();         
    ss >> std::ws;        
}
bool parse_tsp_file(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in)
    {
        std::cerr << "Nie mogę otworzyć " << filename << "\n";
        return false;
    }

    std::string line, key, edge_type = "";
    bool reading_coords = false;

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
                std::transform(edge_type.begin(), edge_type.end(),
                               edge_type.begin(), ::toupper);
            }
            else if (key == "NODE_COORD_SECTION")
            {
                if (num_cities == 0)
                {
                    std::cerr << "Brak DIMENSION przed NODE_COORD_SECTION\n";
                    return false;
                }
                cities_nodes.assign(num_cities, {});
                reading_coords = true;
                for (int i = 0; i < num_cities; ++i)
                {
                    int id;
                    double x, y;
                    if (!(in >> id >> x >> y))
                    {
                        std::cerr << "Błąd współrzędnych\n";
                        return false;
                    }
                    cities_nodes[id - 1] = {id, x, y};
                }
                reading_coords = false;
            }
            else if (key == "EOF")
                break;
        }
    }

    if (cities_nodes.size() != static_cast<size_t>(num_cities))
    {
        std::cerr << "Nie wczytano wszystkich miast\n";
        return false;
    }

    distance_matrix.assign(num_cities, std::vector<double>(num_cities));
    heuristic_matrix.assign(num_cities, std::vector<double>(num_cities));

    for (int i = 0; i < num_cities; ++i)
    {
        for (int j = 0; j < num_cities; ++j)
            if (i != j)
            {
                double d = calc_euc2d(cities_nodes[i], cities_nodes[j]);
                distance_matrix[i][j] = d;
                heuristic_matrix[i][j] = 1.0 / d;
            }
    }
    return true;
}

void initialize_aco_structs()
{
    pheromone_matrix.assign(num_cities, std::vector<double>(num_cities, initial_pheromone));
}

int select_next_city(int current, const Ant &ant)
{
    std::vector<int> cand;
    cand.reserve(num_cities);
    std::vector<double> prob;
    prob.reserve(num_cities);
    double sum = 0.0;
    for (int j = 0; j < num_cities; ++j)
        if (!ant.visited[j])
        {
            double p = std::pow(pheromone_matrix[current][j], alpha) * std::pow(heuristic_matrix[current][j], beta);
            cand.push_back(j);
            prob.push_back(p);
            sum += p;
        }
    if (cand.empty())
        return -1;
    std::uniform_real_distribution<> dist(0.0, sum);
    double r = dist(tls_rng), acc = 0.0;
    for (size_t k = 0; k < cand.size(); ++k)
    {
        acc += prob[k];
        if (r <= acc)
            return cand[k];
    }
    return cand.back();
}

void construct_tour(Ant &ant)
{
    ant.reset();
    ant.initialize(num_cities);
    std::uniform_int_distribution<> start_dist(0, num_cities - 1);
    int current = start_dist(tls_rng);
    ant.tour.push_back(current);
    ant.visited[current] = true;

    for (int step = 1; step < num_cities; ++step)
    {
        int next = select_next_city(current, ant);
        ant.tour.push_back(next);
        ant.visited[next] = true;
        ant.tour_length += distance_matrix[current][next];
        current = next;
    }
    ant.tour_length += distance_matrix[current][ant.tour.front()];
}

void update_pheromones(const std::vector<int> &best_tour, double best_len)
{
    for (int i = 0; i < num_cities; ++i)
        for (int j = 0; j < num_cities; ++j)
            pheromone_matrix[i][j] *= (1.0 - evaporation_rate);

    double dep = Q / best_len;
    for (size_t k = 0; k < best_tour.size() - 1; ++k)
    {
        int u = best_tour[k], v = best_tour[k + 1];
        pheromone_matrix[u][v] += dep;
        pheromone_matrix[v][u] += dep;
    }
    int u = best_tour.back(), v = best_tour.front();
    pheromone_matrix[u][v] += dep;
    pheromone_matrix[v][u] += dep;
}

void run_aco(std::vector<int> &global_best_tour, double &global_best_len)
{
    global_best_len = std::numeric_limits<double>::max();
    std::vector<Ant> ants(num_ants);
    for (auto &a : ants)
        a.initialize(num_cities);

    for (int it = 0; it < max_iterations; ++it)
    {
// ---- parallel section: each ant builds its tour --------------------------------
#pragma omp parallel
        {
            init_thread_rng(); 
#pragma omp for schedule(static)
            for (int i = 0; i < num_ants; ++i)
            {
                construct_tour(ants[i]);
            }
        }
        // find best ant in this iteration
        for (const auto &a : ants)
            if (a.tour_length < global_best_len)
            {
                global_best_len = a.tour_length;
                global_best_tour = a.tour;
            }
        update_pheromones(global_best_tour, global_best_len);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Błąd: Podaj przynajmniej jeden plik .tsp jako argument." << std::endl;
        std::cerr << "Użycie: " << argv[0] << " <plik1.tsp> [plik2.tsp] ..." << std::endl;
        return 1;
    }

    std::ofstream log("results_parallel.txt", std::ios::app);
    log << "NAME" << '\t'
        << "BEST LEN" << '\t'
        << "TIME" << std::endl;

    for (int i = 1; i < argc; ++i)
    {
        const std::string filename = argv[i];
        std::cout << "\n[" << i << "/" << argc - 1 << "] Przetwarzanie pliku: " << filename << "\n";

        if (!parse_tsp_file(filename))
            continue; 

        initialize_aco_data_structures();

        std::vector<int> best_tour;
        double best_len;
        auto t0 = std::chrono::high_resolution_clock::now();
        run_aco(best_tour, best_len);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "Najlepsza długość: "
                  << std::fixed << std::setprecision(2) << best_len
                  << "   czas: " << ms << " ms\n";

        if (log)
        {
            log << tsp_name << '\t'
                << std::fixed << std::setprecision(2) << best_len << '\t'
                << ms << " ms" << std::endl;
        }
    }

    return 0;
}
