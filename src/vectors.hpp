#ifndef VECTORS_HPP
#define VECTORS_HPP


#include <numeric>
#include <vector>
#include <mpi.h>
#include <map>
#include <math.h>
#include <stdexcept>
//#include <functional>
#include <fstream>
#include <tuple>
#include <algorithm>
#include <string>
#include <sstream>
#include <random>
#include <ranges>
#include <chrono>
#include <limits>
#include <execution>
#include <iterator>
#include <iostream>
#include <unordered_map>
#include <experimental/simd>

#define RESTRICT

double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b);
double euclidean_distance(const double* a, const double* b, std::size_t n);
double scalar_product(const std::vector<double>& a, const std::vector<double>& b);
double scalar_product(const double* a, const double* b, std::size_t n);
bool all_equal_size(const std::vector<std::vector<double>>& vec);
bool all_equal_size(const std::vector<std::vector<std::string>>& vec);
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points);
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points, const double gamma);
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points, const double coef0, const double degree);

double* getKernelMatrix(const double *x, const std::size_t n, const std::size_t m);
double* getKernelMatrix(const double *x, const std::size_t n, const std::size_t m, const double gamma);
double* getKernelMatrix(const double *x, const std::size_t n, const std::size_t m, const double coef0, const double degree);


double* kernel_polynomial_blocked(const double* RESTRICT X, size_t n, size_t d, double coef0, double degree, size_t B = 128);

size_t genInt(const size_t n, const size_t k);

double f(const std::vector<double>& d, const std::vector<int>& y, const std::vector<double>& a, const double b);
double f(const double* k, const double* a, const double b, const std::size_t *idp, const std::size_t *idn, std::size_t np, std::size_t nn);

std::vector<std::vector<std::string>> readCSV(const std::string& filename);
bool validateAllButLastAsDouble(const std::vector<std::string>& vec);
double getAccuracy(const std::vector<std::string>& a, const std::vector<std::string>& b);
double getAccuracy(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b);
void trim_whitespace_inplace(std::string& s);
void trimStringVectorSpaces(std::vector<std::string>& vec);
void trimStringMatrixSpaces(std::vector<std::vector<std::string>> &matr);
std::vector<double> convertToDoubleVector(const std::vector<std::string>& strVec);
std::vector<double> convertToDoubleVector(const std::vector<std::string>& strVec, const std::size_t pos);
std::vector<std::vector<double>> convertToDoubleMatrix(const std::vector<std::vector<std::string>>& strVec);
std::vector<std::vector<double>> convertToDoubleMatrix(const std::vector<std::vector<std::string>>& strVec, const std::size_t pos);

std::tuple<double*, std::size_t, std::size_t> convertToDoubleArray(const std::vector<std::vector<double>>& double_m);

std::tuple<double*, std::size_t, std::size_t> convertToDoubleArray(const std::vector<std::vector<std::string>>& strMatr);
std::tuple<double*, std::size_t, std::size_t> convertToDoubleArray(const std::vector<std::vector<std::string>>& strMatr, const std::size_t pos);

void removeFirstElement(std::vector<std::string>& vec);
void removeFirstElement(std::vector<std::vector<std::string>>& vec);
std::vector<std::string> getLastTable(const std::vector<std::vector<std::string>>& matr);

void autoscaling(std::vector<std::vector<double>>& matr);
void autoscaling(double* const x, const std::size_t n, const std::size_t m);

std::vector<int> convertToIntVec(const std::vector<std::string>& vec, std::string& first);
void to_lower(std::string& str);
void printResults(const std::vector<std::string>& vec);
void printResults(const std::vector<std::string>& vec, const double acc);
void printResults(const std::string &filename, const std::vector<std::string>& vec);
void printResults(const std::string &filename, const std::vector<std::string>& vec, const double acc);
std::string mostFrequentString(const std::vector<std::string>& input);
std::string mostFrequentString(const std::vector<std::string>& input, const size_t n);
std::size_t mostFrequentInt(const std::vector<std::size_t>& input);

double getBalancedAccuracy(const std::vector<std::string>& refenced, const std::vector<std::string>& obtained);
std::string getMostBalancedStr(const std::vector<std::pair<std::string, double>>& vec);

std::unordered_map<std::string, std::vector<std::size_t>> get_data_map(const std::vector<std::string>& y);
std::map<std::string, std::vector<std::size_t>> get_ordered_data_map(const std::vector<std::string>& y);

std::vector<std::tuple<std::string, std::vector<std::size_t>>> get_data_vector(const std::vector<std::string>& y);
std::vector<std::tuple<std::string, std::vector<std::size_t>>> get_ordered_data_vector(const std::vector<std::string>& y);

std::unordered_map<std::string, std::vector<std::size_t>> get_data_map(const std::vector<std::vector<std::string>>& str_matr);
std::vector<std::tuple<std::string, std::vector<std::size_t>>> get_data_vector(const std::vector<std::vector<std::string>>& str_matr);

std::size_t choose_random(const std::vector<std::size_t>& data);

std::tuple<char*, std::size_t> vec_to_chars(const std::vector<std::string>& vec);
std::vector<std::string> chars_to_str_vec(char *str, size_t n);
std::vector<std::string> chars_to_str_vec(char *str);
std::vector<std::string> chars_to_str_vec(const char *str);

double getAccuracy(const char* ac, const char* bc);
double getAccuracy(const char* ac, const std::vector<std::string>& b);

double getBalancedAccuracy(const char* referenced_c, const std::vector<std::string>& obtained);


double* MPI_Getkernelmatrix(const double *x, const std::size_t n, const std::size_t m);
double* MPI_Getkernelmatrix(const double *x, const std::size_t n, const std::size_t m, const double gamma);
double* MPI_Getkernelmatrix(const double *x, const std::size_t n, const std::size_t m, const double coef0, const double degree);

#endif
