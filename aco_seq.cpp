// ANT Colony Optimization for TSP Single-threaded version
//g++ aco_seq.cpp -o aco_seq -std=c++17 -O2   
//./aco_seq

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
std::vector<std::vector<double>> pheromone_matrix;
std::vector<std::vector<double>> heuristic_matrix;
std::string tsp_name;
namespace fs = std::filesystem;
// Parametry ACO
int num_ants = 300;
double alpha = 1.0;
double beta = 3.0;
double evaporation_rate = 0.25; // ρ
double Q = 100.0;               // ilość feromonu
int max_iterations = 750;
double initial_pheromone = 0.1;

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

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

// ---------------------------  DEKLARACJE FUNKCJI  -----------------------------
bool parse_tsp_file(const std::string &filename);
void initialize_aco_data_structures();
void run_aco(std::vector<int> &best_tour_overall, double &best_length);
void construct_tour_for_ant(Ant &ant);
int select_next_city(int current_city_idx, const Ant &ant);
void update_pheromones(const std::vector<int> &tour, double length);
inline void eat_ws_and_colon(std::stringstream &ss)
{
    ss >> std::ws;        
    if (ss.peek() == ':') 
        ss.get();         
    ss >> std::ws;        
}
// ---------------------------  POMOCNICZE ODLEGŁOŚCI  --------------------------
double calculate_euc_2d_distance(const City &a, const City &b)
{
    return std::round(std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)));
}

// -------------------------------- PARSOWANIE TSPLIB ---------------------------
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
                        std::cerr << "Błąd czytania współrzędnych\n";
                        return false;
                    }
                    if (id < 1 || id > num_cities)
                    {
                        std::cerr << "Niepoprawny ID miasta " << id << "\n";
                        return false;
                    }
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
    {
        std::cerr << "Nie wczytano wszystkich miast\n";
        return false;
    }

    distance_matrix.assign(num_cities, std::vector<double>(num_cities));
    heuristic_matrix.assign(num_cities, std::vector<double>(num_cities));

    for (int i = 0; i < num_cities; ++i)
    {
        for (int j = 0; j < num_cities; ++j)
        {
            if (i == j)
                continue;
            double d;
            if (edge_type == "EUC_2D" || edge_type == ":" || edge_type.empty())
                d = calculate_euc_2d_distance(cities_nodes[i], cities_nodes[j]);
            else
            {
                std::cerr << "EDGE_WEIGHT_TYPE " << edge_type << " nieobsługiwany\n";
                return false;
            }
            distance_matrix[i][j] = d;
            heuristic_matrix[i][j] = 1.0 / d;
        }
    }
    return true;
}

// --------------------------  ALGORYTM MRÓWKOWY  ------------------------------
void initialize_aco_data_structures()
{
    pheromone_matrix.assign(num_cities, std::vector<double>(num_cities, initial_pheromone));
}

int select_next_city(int current, const Ant &ant)
{
    std::vector<int> candidates;
    std::vector<double> probs;
    double sum = 0.0;
    for (int j = 0; j < num_cities; ++j)
        if (!ant.visited[j])
        {
            double tau = pheromone_matrix[current][j];
            double eta = heuristic_matrix[current][j];
            double p = std::pow(tau, alpha) * std::pow(eta, beta);
            candidates.push_back(j);
            probs.push_back(p);
            sum += p;
        }
    if (candidates.empty())
        return -1;

    std::uniform_real_distribution<> dist(0.0, sum);
    double r = dist(rng), acc = 0.0;
    for (size_t k = 0; k < candidates.size(); ++k)
    {
        acc += probs[k];
        if (r <= acc)
            return candidates[k];
    }
    return candidates.back();
}

void construct_tour_for_ant(Ant &ant)
{
    ant.reset();
    ant.initialize(num_cities);
    std::uniform_int_distribution<> start_dist(0, num_cities - 1);
    int current = start_dist(rng);
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

    double deposit = Q / best_len;
    for (size_t k = 0; k < best_tour.size() - 1; ++k)
    {
        int u = best_tour[k], v = best_tour[k + 1];
        pheromone_matrix[u][v] += deposit;
        pheromone_matrix[v][u] += deposit;
    }
    int u = best_tour.back(), v = best_tour.front();
    pheromone_matrix[u][v] += deposit;
    pheromone_matrix[v][u] += deposit;
}

void run_aco(std::vector<int> &best_tour, double &best_len)
{
    best_len = std::numeric_limits<double>::max();
    std::vector<Ant> ants(num_ants);
    for (auto &a : ants)
        a.initialize(num_cities);

    for (int it = 0; it < max_iterations; ++it)
    {
        for (auto &a : ants)
            construct_tour_for_ant(a);
        for (const auto &a : ants)
            if (a.tour_length < best_len)
            {
                best_len = a.tour_length;
                best_tour = a.tour;
            }
        update_pheromones(best_tour, best_len);
    }
}

int main()
{
    constexpr char DATA_DIR[] = "data";
    std::vector<fs::path> tsp_files;
    
    for (const auto &e : fs::directory_iterator(DATA_DIR))
        if (e.is_regular_file() && e.path().extension() == ".tsp")
            tsp_files.push_back(e.path());

    std::sort(tsp_files.begin(), tsp_files.end()); // alfabetycznie

    if (tsp_files.empty())
    {
        std::cerr << "Brak plików .tsp w " << DATA_DIR << "\n";
        return 1;
    }

    std::ofstream log("results_tsp.txt", std::ios::app);
    log << "NAME" << '\t'
        << "BEST LEN" << '\t'
        << "TIME" << std::endl;
    for (size_t idx = 0; idx < tsp_files.size(); ++idx)
    {
        const std::string filename = tsp_files[idx].string();
        std::cout << "\n[" << idx + 1 << "/" << tsp_files.size()
                  << "] " << tsp_files[idx].filename() << "\n";

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
