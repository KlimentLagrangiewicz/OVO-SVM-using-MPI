#include "vectors.hpp"


template <class T>
static T simd_euclidean_distance(const T* a, const T* b, const std::size_t n) {
	static_assert(std::is_floating_point<T>::value, "T must be float, double or long double");
	
	if (!a || !b || n == 0) return (T)0;
	
	using namespace std::experimental;
	using simd_t = native_simd<T>;
	constexpr std::size_t width = simd_t::size();
	
	simd_t acc = (T)0;
	std::size_t i = 0;
	
	for (; i + width <= n; i += width) {
		simd_t va(a + i, element_aligned);
		simd_t vb(b + i, element_aligned);
		simd_t d = va - vb;
		acc += d * d;
	}
	T sum = reduce(acc);
	for (; i < n; ++i) {
		const T d = a[i] - b[i];
		sum += d * d;
	}
	
	return std::sqrt(sum);
}


template <class T>
static T simd_scalar_product(const T* a, const T* b, const std::size_t n) {
	static_assert(std::is_floating_point<T>::value, "T must be float, double or long double");
	
	if (!a || !b || n == 0) return (T)0;
	
	using namespace std::experimental;
	using simd_t = native_simd<T>;
	
	constexpr std::size_t width = simd_t::size();
	
	simd_t vsum( T(0) );
	
	size_t i = 0;
	for (; i + width <= n; i += width) {
		simd_t va(a + i, element_aligned_tag{});
		simd_t vb(b + i, element_aligned_tag{});
		vsum += va * vb;
	}
	
	T sum = reduce(vsum);
	
	for (; i < n; ++i)
		sum += a[i] * b[i];
	
	return sum;
}


double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) [[unlikely]] throw std::invalid_argument("Vectors must have the same dimension");
	/*
	return std::sqrt(
		std::transform_reduce(
			std::execution::unseq, 
			a.begin(),
			a.end(),
			b.begin(),
			0.0,
			std::plus<double>(),
			[](double x, double y) {
				double d = x - y;
				return d * d;
			}
		)
	);
	*/
	
	return simd_euclidean_distance(a.data(), b.data(), a.size());
}

/*
double euclidean_distance(const double* a, const double* b, std::size_t n) {
	double res = 0.0;
	for (; n > 0; --n) {
		const double d = *a - *b;
		res += d * d;
		++a;
		++b;
	}
	return std::sqrt(res);
}
*/

double euclidean_distance(const double* a, const double* b, std::size_t n) {
	/*
	return std::sqrt(
		std::transform_reduce(
			std::execution::unseq, 
			a, 
			a + n, 
			b, 
			0.0,
			std::plus<double>(),
			[](double x, double y) {
				double d = x - y;
				return d * d;
			}
		)
	);
	*/
	return simd_euclidean_distance(a, b, n);
}

double scalar_product(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) [[unlikely]] throw std::invalid_argument("Vectors must have the same dimension");
	/*
	return std::transform_reduce(
			std::execution::unseq,
			a.begin(),
			a.end(),
			b.begin(),
			0.0,
			std::plus<double>(),
			[](const double x, const double y) {
				return x * y;
			}
	);
	*/
	return simd_scalar_product(a.data(), b.data(), a.size());
}

double scalar_product(const double* a, const double* b, std::size_t n) {
	/*
	return std::transform_reduce(
			std::execution::unseq, 
			a, 
			a + n, 
			b, 
			0.0,
			std::plus<double>(),
			[](const double x, const double y) {
				return x * y;
			}
	);
	*/
	return simd_scalar_product(a, b, n);
	
}

bool all_equal_size(const std::vector<std::vector<double>>& vec) {
	if (vec.empty()) return true;
	const auto expected = vec.front().size();
	return std::all_of(
		vec.begin() + 1,
		vec.end(),
		[&expected](const std::vector<double>& v) { return v.size() == expected; }
	);
}

bool all_equal_size(const std::vector<std::vector<std::string>>& vec) {
	if (vec.empty()) return true;
	const auto expected = vec.front().size();
	return std::all_of(
		vec.begin() + 1,
		vec.end(),
		[&expected](const std::vector<std::string>& v) { return v.size() == expected; }
	);
}

// Liner Kernel
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points) {
	if (points.empty()) [[unlikely]] return {};
	if (!all_equal_size(points)) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	const std::vector<std::vector<double>>::size_type m = points.size();
	std::vector<std::vector<double>> matrix(m, std::vector<double>(m)); 
	for (std::vector<double>::size_type i = 0; i < m; ++i) {
		matrix[i][i] = scalar_product(points[i], points[i]);
		for (std::vector<double>::size_type j = 0; j < i; ++j) {
			const double d = scalar_product(points[i], points[j]);
			matrix[i][j] = d;
			matrix[j][i] = d;
		}
	}
	return matrix;
}

// Liner Kernel
double* getKernelMatrix(const double *x, const std::size_t n, const std::size_t m) {
	if (!x) [[unlikely]] throw std::invalid_argument("Entered null array of points");
	double *matrix = new double [n * n];
	if (!matrix) [[unlikely]] throw std::runtime_error("Memory allocation error for linear kernel");
	for (std::size_t i = 0; i < n; ++i) {
		const double* const x_i = x + i * m;
		double* const m_i = matrix + i * n;
		
		for (std::size_t j = 0; j < i; ++j) {
			m_i[j] = scalar_product(x_i, x + j * m, m);
			matrix[j * n + i] = m_i[j];
		}
		m_i[i] = scalar_product(x_i, x_i, m);
	}
	return matrix;
}

double* MPI_Getkernelmatrix(const double *x, const std::size_t n, const std::size_t m) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (rank == 0) {
		if (!x) throw std::invalid_argument("Entered null array of points");
	}
	
	std::vector<int> row_counts(size, 0);
	std::vector<int> row_displs(size, 0);
	
	int base_rows = n / size;
	int remainder = n % size;
	    
	for (int i = 0; i < size; ++i) {
		row_counts[i] = base_rows + (i < remainder ? 1 : 0);
		if (i > 0) row_displs[i] = row_displs[i-1] + row_counts[i-1];
	}
	
	
	int start_row = row_displs[rank];
	int local_rows = row_counts[rank];
	
	double* local_matrix = new double[local_rows * n];
	
	for (int local_i = 0; local_i < local_rows; ++local_i) {
		std::size_t i = start_row + local_i;
		const double* x_i = x + i * m;
		double* m_i = local_matrix + local_i * n;
		
		for (std::size_t j = 0; j < n; ++j) {
			const double* x_j = x + j * m;
			m_i[j] = scalar_product(x_i, x_j, m);
		}
	}
	
	std::vector<int> recvcounts(size);
	std::vector<int> displs(size);
	
	for (int i = 0; i < size; ++i) {
		recvcounts[i] = row_counts[i] * n;
		displs[i] = row_displs[i] * n;
	}
	
	double* matrix = new double[n * n];
	MPI_Allgatherv(local_matrix, local_rows * n, MPI_DOUBLE, matrix, recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
	
	
	delete[] local_matrix;
	
	return matrix;
}

// RBF Kernel
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points, const double gamma) {
	if (points.empty()) [[unlikely]] return {};
	if (!all_equal_size(points)) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	const std::vector<std::vector<double>>::size_type m = points.size();
	std::vector<std::vector<double>> matrix(m, std::vector<double>(m)); 
	for (std::vector<double>::size_type i = 0; i < m; ++i) {
		matrix[i][i] = 1.0;
		for (std::vector<double>::size_type j = 0; j < i; ++j) {
			const double ed = euclidean_distance(points[i], points[j]);
			const double d = std::exp(-gamma * ed * ed);
			matrix[i][j] = d;
			matrix[j][i] = d;
		}
	}
	return matrix;
}

// RBF Kernel
double* getKernelMatrix(const double *x, const std::size_t n, const std::size_t m, const double gamma) {
	if (!x) [[unlikely]] throw std::invalid_argument("Entered null array of points");
	double *matrix = new double [n * n];
	if (!matrix) [[unlikely]] throw std::runtime_error("Memory allocation error for linear kernel");
	
	
	for (std::size_t i = 0; i < n; ++i) {
		const double* const x_i = x + i * m;
		double* const m_i = matrix + i * n;
		m_i[i] = 1.0;
		
		for (std::size_t j = 0; j < i; ++j) {
			const double ed = euclidean_distance(x_i, x + j * m, m);
			const double d = std::exp(-gamma * ed * ed);
			m_i[j] = d;
			matrix[j * n + i] = d;
		}
	}
	
	return matrix;
}

double* MPI_Getkernelmatrix(const double *x, const std::size_t n, const std::size_t m, const double gamma) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (rank == 0) {
		if (!x) throw std::invalid_argument("Entered null array of points");
	}
	
	std::vector<int> row_counts(size, 0);
	std::vector<int> row_displs(size, 0);
	
	int base_rows = n / size;
	int remainder = n % size;
	    
	for (int i = 0; i < size; ++i) {
		row_counts[i] = base_rows + (i < remainder ? 1 : 0);
		if (i > 0) row_displs[i] = row_displs[i-1] + row_counts[i-1];
	}
	
	
	int start_row = row_displs[rank];
	int local_rows = row_counts[rank];
	
	double* local_matrix = new double[local_rows * n];
	
	for (int local_i = 0; local_i < local_rows; ++local_i) {
		std::size_t i = start_row + local_i;
		const double* x_i = x + i * m;
		double* m_i = local_matrix + local_i * n;
		
		for (std::size_t j = 0; j < n; ++j) {
			const double* x_j = x + j * m;
			const double ed = euclidean_distance(x_i, x_j, m);
			m_i[j] = std::exp(-gamma * ed * ed);
		}
	}
	
	std::vector<int> recvcounts(size);
	std::vector<int> displs(size);
	
	for (int i = 0; i < size; ++i) {
		recvcounts[i] = row_counts[i] * n;
		displs[i] = row_displs[i] * n;
	}
	
	double* matrix = new double[n * n];
	MPI_Allgatherv(local_matrix, local_rows * n, MPI_DOUBLE, matrix, recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
	
	
	delete[] local_matrix;
	
	return matrix;
}


// Poly Kernel
std::vector<std::vector<double>> getKernelMatrix(const std::vector<std::vector<double>>& points, const double coef0, const double degree) {
	if (points.empty()) [[unlikely]] return {};
	if (!all_equal_size(points)) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	const std::vector<std::vector<double>>::size_type m = points.size();
	std::vector<std::vector<double>> matrix(m, std::vector<double>(m)); 
	for (std::vector<std::vector<double>>::size_type i = 0; i < m; ++i) {
		for (std::vector<std::vector<double>>::size_type j = 0; j < i; ++j) {
			const double sp = scalar_product(points[i], points[j]);
			const double d = std::pow(sp + coef0, degree);
			matrix[i][j] = d;
			matrix[j][i] = d;
		}
		const double sp = scalar_product(points[i], points[i]);
		const double d = std::pow(sp + coef0, degree);
		matrix[i][i] = d;
	}
	return matrix;
}


double* getKernelMatrix(const double *x, const std::size_t n, const std::size_t m, const double coef0, const double degree) {
	if (!x) [[unlikely]] throw std::invalid_argument("Entered null array of points");
	double *matrix = new double [n * n];
	if (!matrix) [[unlikely]] throw std::runtime_error("Memory allocation error for linear kernel");
	for (std::size_t i = 0; i < n; ++i) {
		const double* const x_i = x + i * m;
		double* const m_i = matrix + i * n;
		
		for (std::size_t j = 0; j < i; ++j) {
			const double sp = scalar_product(x_i, x + j * m, m);
			const double d = std::pow(sp + coef0, degree);
			m_i[j] = d;
			matrix[j * n + i] = d;
		}
		
		const double sp = scalar_product(x_i, x_i, m);
		const double d = std::pow(sp + coef0, degree);
		m_i[i] = d;
	}
	return matrix;
}


double* MPI_Getkernelmatrix(const double *x, const std::size_t n, const std::size_t m, const double coef0, const double degree) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (rank == 0) {
		if (!x) throw std::invalid_argument("Entered null array of points");
	}
	
	std::vector<int> row_counts(size, 0);
	std::vector<int> row_displs(size, 0);
	
	int base_rows = n / size;
	int remainder = n % size;
	    
	for (int i = 0; i < size; ++i) {
		row_counts[i] = base_rows + (i < remainder ? 1 : 0);
		if (i > 0) row_displs[i] = row_displs[i-1] + row_counts[i-1];
	}
	
	
	int start_row = row_displs[rank];
	int local_rows = row_counts[rank];
	
	double* local_matrix = new double[local_rows * n];
	
	for (int local_i = 0; local_i < local_rows; ++local_i) {
		std::size_t i = start_row + local_i;
		const double* x_i = x + i * m;
		double* m_i = local_matrix + local_i * n;
		
		for (std::size_t j = 0; j < n; ++j) {
			const double* x_j = x + j * m;
			const double sp = scalar_product(x_i, x_j, m);
			m_i[j] = std::pow(sp + coef0, degree);
		}
	}
	
	std::vector<int> recvcounts(size);
	std::vector<int> displs(size);
	
	for (int i = 0; i < size; ++i) {
		recvcounts[i] = row_counts[i] * n;
		displs[i] = row_displs[i] * n;
	}
	
	double* matrix = new double[n * n];
	MPI_Allgatherv(local_matrix, local_rows * n, MPI_DOUBLE, matrix, recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
	
	
	delete[] local_matrix;
	
	return matrix;
}

double* kernel_polynomial_blocked(const double* RESTRICT X, size_t n, size_t d, double coef0, double degree, size_t B) {
	auto idx = [n](size_t i, size_t j) { return i * n + j; };
	if (n == 0) return nullptr;
	
	double* const RESTRICT K = new double [n * n];
	
	for (size_t bi = 0; bi < n; bi += B) {
		for (size_t bj = bi; bj < n; bj += B) {
			const size_t iMax = std::min(bi + B, n);
			const size_t jMax = std::min(bj + B, n);
			
			for (size_t i = bi; i < iMax; ++i) {
				const double* xi = X + i * d;
				const size_t j0 = (bj == bi) ? i : bj; // в диагональной плитке j начинается с i
				for (size_t j = j0; j < jMax; ++j) {
					const double* xj = X + j * d;
					const double gij = scalar_product(xi, xj, d);
					const double v   = gij + coef0;
					const double kij = std::pow(v, degree);
					
					K[idx(i, j)] = kij;
					K[idx(j, i)] = kij;
				}
			}
		}
	}
	return K;
}

size_t genInt(const size_t n, const size_t k) {
	std::mt19937 gen((std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()) % 1000000000).count());
	if (k == 0) {
		std::uniform_int_distribution<size_t> distrib(1, n - 1);
		return distrib(gen);
	}
	if (k == n - 1) {
		std::uniform_int_distribution<size_t> distrib(0, n - 2);
		return distrib(gen);
	}
	std::uniform_int_distribution<size_t> distrib(0, n - 1);
	const size_t r = distrib(gen);
	return r == k ? r + 1 : r;
}

double f(const std::vector<double>& d, const std::vector<int>& y, const std::vector<double>& a, const double b) {
	const size_t n = d.size();
	if (n != y.size() || n != a.size()) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	const double* __restrict d_ptr = d.data();
	const int* __restrict y_ptr = y.data();
	const double* __restrict a_ptr = a.data();
	double res = 0.0;
	for (size_t i = 0; i < n; ++i) {
		res += y_ptr[i] * a_ptr[i] * d_ptr[i];
	}
	return res + b;
}

double f(const double* k, const double* a, const double b, const std::size_t *idp, const std::size_t *idn, std::size_t np, std::size_t nn) {
	if (!k || !idp || !a || !idn) [[unlikely]] throw std::invalid_argument("One or more of input arrays are null");
	double res = 0.0;
	
	for (std::size_t i = 0; i < np; ++i) {
		res += a[i] * k[idp[i]];
	}
	
	for (std::size_t i = 0; i < nn; ++i) {
		res -= a[np + i] * k[idn[i]];
	}
	
	return res + b;
}

std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
	std::vector<std::vector<std::string>> result;
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
		return {};
	}
	std::string line;
	while (std::getline(file, line)) {
		std::vector<std::string> row;
		std::stringstream ss(line);
		std::string cell;
		while (std::getline(ss, cell, ',')) row.push_back(cell);
		result.push_back(row);
	}
	file.close();
	return result;
}

bool validateAllButLastAsDouble(const std::vector<std::string>& vec) {
	if (vec.empty()) return false;
	if (vec.size() == 1) return true;
	for (size_t i = 0; i < vec.size() - 1; ++i) {
		std::istringstream iss(vec[i]);
		double value;
		if (!(iss >> value) || (iss >> std::ws && !iss.eof())) {
			return false;
		}
	}
	return true;
}

double getAccuracy(const std::vector<std::string>& a, const std::vector<std::string>& b) {
	const auto n = a.size();
	if (n != b.size())[[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	if (n == 0) return 0.0;
	const auto count = std::inner_product(
		a.begin(), 
		a.end(),
		b.begin(),
		0ULL,
		std::plus<>(),
		[](const std::string& x, const std::string& y) { return x == y ? 1ULL : 0ULL; }
	);
	return static_cast<double>(count) / static_cast<double>(n);
}

double getAccuracy(const std::vector<std::size_t>& a, const std::vector<std::size_t>& b) {
	const auto n = a.size();
	if (n != b.size()) [[unlikely]] throw std::invalid_argument("All points must have the same dimension");
	if (n == 0) return 0.0;
	const auto count = std::inner_product(
		a.begin(), 
		a.end(),
		b.begin(),
		0ULL,
		std::plus<>(),
		[](const std::size_t& x, const std::size_t& y) { return x == y ? 1ULL : 0ULL; }
	);
	return static_cast<double>(count) / static_cast<double>(n);
}

void trim_whitespace_inplace(std::string& s) {
	static constexpr const char* const whitespace = " \t\n\r\f\v";
	if (s.empty()) return;
	const size_t first = s.find_first_not_of(whitespace);
	if (first == std::string::npos) {
		s.clear();
		return;
	}
	const size_t last = s.find_last_not_of(whitespace);
	const size_t len = last - first + 1;
	if (first > 0 || last < s.size() - 1) {
		s.assign(s, first, len);
	}
}

void trimStringVectorSpaces(std::vector<std::string>& vec) {
	std::for_each(vec.begin(), vec.end(), [](std::string& s) { trim_whitespace_inplace(s); });
}

void trimStringMatrixSpaces(std::vector<std::vector<std::string>> &matr) {
	std::for_each(matr.begin(), matr.end(), [] (std::vector<std::string>& vec) { trimStringVectorSpaces(vec); });
}

std::vector<double> convertToDoubleVector(const std::vector<std::string>& strVec) {
	std::vector<double> result;
	result.reserve(strVec.size());
	try {
		for (const auto& str : strVec) {
			std::size_t pos = 0;
			double value = std::stod(str, &pos);
			if (pos != str.size()) {
				throw std::invalid_argument("Invalid characters in element: '" + str + "'");
			}
			result.push_back(value);
		}
	} catch (const std::exception& e) {
		throw std::runtime_error(std::string("Conversion failed: ") + e.what());
	}
	return result;
}

static double convertToDouble(const std::string &str) {
	std::size_t p = 0;
	double value = std::stod(str, &p);
	if (p != str.size()) {
		throw std::invalid_argument("Invalid characters in element '" + str + "'");
	}
	return value;
}


std::vector<double> convertToDoubleVector(const std::vector<std::string>& strVec, const std::size_t pos) {
	if (pos == 0 || strVec.empty()) return {};
	std::vector<double> result;
	result.reserve(pos - 1);
	try {
		for (size_t i = 0; i < pos; ++i) {
			double value = convertToDouble(strVec[i]);

			result.push_back(value);
		}
	} catch (const std::exception& e) {
		throw std::runtime_error(std::string("Conversion failed: ") + e.what());
	}
	return result;
}


std::vector<std::vector<double>> convertToDoubleMatrix(const std::vector<std::vector<std::string>>& strVec) {
	std::vector<std::vector<double>> res;
	res.reserve(strVec.size());
	std::transform(strVec.begin(), strVec.end(), std::back_inserter(res), [](const std::vector<std::string>& s) { return convertToDoubleVector(s); });
	return res;
}

std::vector<std::vector<double>> convertToDoubleMatrix(const std::vector<std::vector<std::string>>& strVec, const std::size_t pos) {
	std::vector<std::vector<double>> res;
	res.reserve(strVec.size());
	std::transform(strVec.begin(), strVec.end(), std::back_inserter(res), [pos](const std::vector<std::string>& s) { return convertToDoubleVector(s, pos); });
	return res;
}

std::tuple<double*, std::size_t, std::size_t> convertToDoubleArray(const std::vector<std::vector<std::string>>& strMatr) {
	if (strMatr.empty()) [[unlikely]] throw std::invalid_argument("Input srting matrix is empty");
	if (!all_equal_size(strMatr)) [[unlikely]] throw std::invalid_argument("Incorrect size of input srting matrix");//
	const size_t n = strMatr.size(), m = strMatr[0].size();
	double *arr = new double [n * m];
	double *ptr = arr;
	try {
		for (const auto& strVec : strMatr) {
			for (const auto& str : strVec) {
				*ptr = convertToDouble(str);
				++ptr;
			}
		}
	} catch (const std::exception& e) {
		throw std::runtime_error(std::string("Conversion failed: ") + e.what());
	}
	return {arr, n, m};
}

std::tuple<double*, std::size_t, std::size_t> convertToDoubleArray(const std::vector<std::vector<std::string>>& strMatr, const std::size_t pos) {
	if (strMatr.empty()) [[unlikely]] throw std::invalid_argument("Input srting matrix is empty");
	if (!all_equal_size(strMatr)) [[unlikely]] throw std::invalid_argument("Incorrect size of input srting matrix");//
	if (pos >= strMatr[0].size()) [[unlikely]] throw std::invalid_argument("Invalid column number to skip");
	
	const size_t n = strMatr.size(), m = strMatr[0].size() - 1;
	if (m == 0) [[unlikely]] throw std::invalid_argument("Can't transform matrix with one column without that column");
	
	double *arr = new double [n * m];
	double *ptr = arr;
	try {
		for (const auto& strVec : strMatr) {
			for (std::vector<std::string>::size_type i = 0; i < pos; ++i) {
				*ptr = convertToDouble(strVec[i]);
				++ptr;
			}
			for (std::vector<std::string>::size_type i = pos + 1; i < strVec.size(); ++i) {
				*ptr = convertToDouble(strVec[i]);
				++ptr;
			}
		}
	} catch (const std::exception& e) {
		throw std::runtime_error(std::string("Conversion failed: ") + e.what());
	}
	return {arr, n, m};
}

std::tuple<double*, std::size_t, std::size_t> convertToDoubleArray(const std::vector<std::vector<double>>& double_m) {
	if (double_m.empty()) [[unlikely]] throw std::invalid_argument("Input srting matrix is empty");
	if (!all_equal_size(double_m)) [[unlikely]] throw std::invalid_argument("Incorrect size of input double matrix");
	
	const size_t n = double_m.size(), m = double_m[0].size();
	
	double *arr = new double [n * m];
	double *ptr = arr;
	try {
		for (const auto& vec : double_m) {
			for (const auto& d_v : vec) {
				*ptr = d_v;
				++ptr;
			}
		}
	} catch (const std::exception& e) {
		throw std::runtime_error(std::string("Conversion failed: ") + e.what());
	}
	
	return {arr, n, m};
}

void removeFirstElement(std::vector<std::string>& vec) {
	if (!vec.empty()) vec.erase(vec.begin());
}

void removeFirstElement(std::vector<std::vector<std::string>>& vec) {
	if (!vec.empty()) vec.erase(vec.begin());
}

std::vector<std::string> getLastTable(const std::vector<std::vector<std::string>>& matr) {
	std::vector<std::string> res;
	res.reserve(matr.size());
	std::ranges::transform(
		matr,
		std::back_inserter(res),
		[](const auto& vec) -> std::string { return vec.empty() ? std::string{} : vec.back(); }
	);
	return res;
}

void autoscaling(std::vector<std::vector<double>>& matr) {
	const size_t n = matr.size(), m = matr[0].size();
	std::vector<double> ex(m, 0.0), exx(m, 0.0);
	for (size_t i = 0; i < n; ++i) {
		const auto& row = matr[i];
		for (size_t j = 0; j < m; ++j) {
			ex[j] += row[j];
			exx[j] += row[j] * row[j];
		}
	}
	for (size_t j = 0; j < m; ++j) {
		ex[j] /= n;
		exx[j] /= n;
		double d = std::sqrt(exx[j] - ex[j] * ex[j]);
		if (d == 0.0) d = 1.0;
		exx[j] = 1.0 / d;
	}
	for (size_t i = 0; i < n; ++i) {
		auto& row = matr[i];
		for (size_t j = 0; j < m; ++j) {
			row[j] = (row[j] - ex[j]) * exx[j];
		}
	}
}

void autoscaling(double* const x, const std::size_t n, const std::size_t m) {
	const double* const end = x + n * m;
	for (std::size_t j = 0; j < m; ++j) {
		double sd, Ex = 0.0, Exx = 0.0;
		for (const double *ptr = static_cast<const double*>(x) + j; ptr < end; ptr += m) {
			sd = *ptr;
			Ex += sd;
			Exx += sd * sd;
		}
		Exx /= static_cast<double>(n);
		Ex /= static_cast<double>(n);
		sd = std::abs(Exx - Ex * Ex);
		if (sd == 0.0) sd = 1.0;
		sd = 1.0 / std::sqrt(sd);
		for (double *ptr = x + j; ptr < end; ptr += m) {
			*ptr = (*ptr - Ex) * sd;
		}
	}
}

std::vector<int> convertToIntVec(const std::vector<std::string>& vec, std::string & first) {
	if (vec.empty()) return {};
	std::vector<int> res;
	res.reserve(vec.size());
	std::ranges::transform(
		vec,
		std::back_inserter(res),
		[first](const std::string& e) { return e == first ? 1 : -1; }
	);
	return res;
}

void to_lower(std::string& str) {
	std::transform(
		str.begin(),
		str.end(),
		str.begin(),
		[](unsigned char c) { return std::tolower(c); }
	);
}

void printResults(const std::vector<std::string>& vec) {
	if (vec.empty()) {
		std::cout << "Result std::vector<std::string> is empty\n";
		return;
	}
	std::cout << "Object: Class\n";
	for (std::vector<std::string>::size_type i = 0; i < vec.size(); ++i) {
		std::cout << "Object[" << i << "]: " << vec[i] << '\n';
	}
	std::cout << '\n';
}

void printResults(const std::vector<std::string>& vec, const double acc) {
	if (vec.empty()) {
		std::cout << "Result std::vector<std::string> is empty\n";
		return;
	}
	std::cout << "Accuracy of SVM-classification = " << acc << '\n';
	std::cout << "Object: Class\n";
	for (std::vector<std::string>::size_type i = 0; i < vec.size(); ++i) {
		std::cout << "Object[" << i << "]: " << vec[i] << '\n';
	}
	std::cout << '\n';
}

void printResults(const std::string &filename, const std::vector<std::string>& vec) {
	std::fstream file;
	file.open(filename, std::ios_base::out);
	if (!file.is_open()) {
		std::runtime_error("Can't open '" + filename + "' file for writing\n");
	}
	if (vec.empty()) {
		file << "Result std::vector<std::string> is empty\n";
		file.close();
		return;
	}
	file << "Object,Class\n";
	for (std::vector<std::string>::size_type i = 0; i < vec.size(); ++i) {
		file << "Object[" << i << "]," << vec[i] << '\n';
	}
	file << '\n';
	file.close();
}

void printResults(const std::string &filename, const std::vector<std::string>& vec, const double acc) {
	std::fstream file;
	file.open(filename, std::ios_base::out);
	if (!file.is_open()) {
		std::runtime_error("Can't open '" + filename + "' file for writing\n");
	}
	if (vec.empty()) {
		file << "Result std::vector<std::string> is empty\n";
		file.close();
		return;
	}
	file << "Accuracy of SVM-classification," << acc << '\n';
	file << "Object,Class\n";
	for (std::vector<std::string>::size_type i = 0; i < vec.size(); ++i) {
		file << "Object[" << i << "]," << vec[i] << '\n';
	}
	file << '\n';
	file.close();
}

std::string mostFrequentString(const std::vector<std::string>& input) {
	if (input.empty()) return "";
	std::unordered_map<std::string, int> frequency_map;
	for (const auto &s: input) {
		++frequency_map[s];
	}
	std::string result;
	int max_count = 0;
	for (const auto& pair: frequency_map) {
		const std::string& s = pair.first;
		int count = pair.second;
		if (count >= max_count) {
			max_count = count;
			result = s;
		}
	}
	return result;
}

std::string mostFrequentString(const std::vector<std::string>& input, const size_t n) {
	if (input.empty()) return "";
	std::unordered_map<std::string, int> frequency_map;
	frequency_map.reserve(n);
	for (const auto &s: input) {
		++frequency_map[s];
	}
	std::string result;
	int max_count = 0;
	for (const auto& pair: frequency_map) {
		const std::string& s = pair.first;
		int count = pair.second;
		if (count >= max_count) {
			max_count = count;
			result = s;
		}
	}
	return result;
}

std::size_t mostFrequentInt(const std::vector<std::size_t>& input) {
	if (input.empty()) throw std::invalid_argument("Input int vector is empty");
	std::unordered_map<size_t, size_t> frequency_map;
	for (const auto &s: input) {
		++frequency_map[s];
	}
	size_t result = frequency_map.begin()->second;
	size_t max_count = frequency_map.begin()->first;
	for (const auto& pair: frequency_map) {
		const size_t& s = pair.first;
		size_t count = pair.second;
		if (count >= max_count) {
			max_count = count;
			result = s;
		}
	}
	return result;
}

double getBalancedAccuracy(const std::vector<std::string>& referenced, const std::vector<std::string>& obtained) {
	const auto n = referenced.size();
	if (obtained.size() != n) throw std::invalid_argument("Vectors must have the same size");
	
	if (n == 0) return 0.0;
	
	std::unordered_map<std::string, std::pair<size_t, size_t>> class_stats;
	
	for (size_t i = 0; i < n; ++i) {
		const auto& true_label = referenced[i];
		auto& stats = class_stats[true_label];
		++stats.first;
		if (true_label == obtained[i]) ++stats.second;
	}
	
	const auto num_classes = class_stats.size();
	
	if (num_classes == 0) return 0.0;
	
	double sum_recall = 0.0;
	for (const auto& [_, counts] : class_stats) {
		sum_recall += static_cast<double>(counts.second) / counts.first;
	}
	
	return sum_recall / num_classes;
}

std::string getMostBalancedStr(const std::vector<std::pair<std::string, double>>& vec) {
	std::unordered_map<std::string, double> map;
	for (const auto &v: vec) {
		map[v.first] += v.second;
	}
	std::string max_key = map.begin()->first;
	double max_val = map.begin()->second;
	for (const auto &pair: map) {
		const double cur_val = pair.second;
		if (cur_val > max_val) {
			max_val = cur_val;
			max_key = pair.first;
		}
	}
	return max_key;
}

std::unordered_map<std::string, std::vector<std::size_t>> get_data_map(const std::vector<std::string>& y) {
	const auto n = y.size();
	if (n == 0) [[unlikely]] throw std::invalid_argument("get_data_map fun error: Empty input vector Y");
	std::unordered_map<std::string, std::vector<size_t>> res;
	for (size_t i = 0; i < n; ++i) {
		if (auto search = res.find(y[i]); search != res.end()) {
			search->second.push_back(i);
		} else {
			res[y[i]].reserve(n);
			res[y[i]].push_back(i);
		}
	}
	for (auto &i: res) {
		i.second.shrink_to_fit();
	}
	return res;
}

std::map<std::string, std::vector<std::size_t>> get_ordered_data_map(const std::vector<std::string>& y) {
	const auto n = y.size();
	if (n == 0) [[unlikely]] throw std::invalid_argument("get_data_map fun error: Empty input vector Y");
	std::map<std::string, std::vector<size_t>> res;
	for (size_t i = 0; i < n; ++i) {
		if (auto search = res.find(y[i]); search != res.end()) {
			search->second.push_back(i);
		} else {
			res[y[i]].reserve(n);
			res[y[i]].push_back(i);
		}
	}
	for (auto &i: res) {
		i.second.shrink_to_fit();
	}
	return res;
}

std::vector<std::tuple<std::string, std::vector<std::size_t>>> get_data_vector(const std::vector<std::string>& y) {
	const auto n = y.size();
	if (n == 0) [[unlikely]] throw std::invalid_argument("get_data_vector fun error: Empty input vector Y");
	const auto &data_map = get_data_map(y);
	std::vector<std::tuple<std::string, std::vector<std::size_t>>> vec;
	vec.reserve(data_map.size());
	for (const auto &el: data_map) {
		vec.push_back({el.first, el.second});
	}

	return vec;
}

std::vector<std::tuple<std::string, std::vector<std::size_t>>> get_ordered_data_vector(const std::vector<std::string>& y) {
	const auto n = y.size();
	if (n == 0) [[unlikely]] throw std::invalid_argument("get_data_vector fun error: Empty input vector Y");
	const auto &data_map = get_ordered_data_map(y);
	std::vector<std::tuple<std::string, std::vector<std::size_t>>> vec;
	vec.reserve(data_map.size());
	for (const auto &el: data_map) {
		vec.push_back({el.first, el.second});
	}

	return vec;
}


std::unordered_map<std::string, std::vector<std::size_t>> get_data_map(const std::vector<std::vector<std::string>>& str_matr) {
	const auto n = str_matr.size();
	if (n == 0) [[unlikely]] throw std::invalid_argument("get_data_map fun error: Empty input matr of strings");
	std::unordered_map<std::string, std::vector<size_t>> res;
	for (size_t i = 0; i < n; ++i) {
		const auto &el = str_matr[i].back();
		if (auto search = res.find(el); search != res.end()) {
			search->second.push_back(i);
		} else {
			res[el].reserve(n);
			res[el].push_back(i);
		}
	}	
	for (auto &i: res) {
		i.second.shrink_to_fit();
	}
	return res;
}

std::vector<std::tuple<std::string, std::vector<std::size_t>>> get_data_vector(const std::vector<std::vector<std::string>>& str_matr) {
	const auto n = str_matr.size();
	if (n == 0) [[unlikely]] throw std::invalid_argument("get_data_vector fun error: Empty input matr of strings");
	
	const auto &data_map = get_data_map(str_matr);
	
	std::vector<std::tuple<std::string, std::vector<std::size_t>>> vec;
	vec.reserve(data_map.size());
	for (const auto &el: data_map) {
		vec.push_back({el.first, el.second});
	}
	
	return vec;
}

std::size_t choose_random(const std::vector<std::size_t>& data) {
	if (data.empty()) [[unlikely]] throw std::invalid_argument("Can't select element from empty array");
	std::mt19937 gen((std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()) % 1000000000).count());
	std::uniform_int_distribution<std::size_t> dist(0, data.size() - 1);
	return data[dist(gen)];
}

std::tuple<char*, std::size_t> vec_to_chars(const std::vector<std::string>& vec) {
	const auto vec_s = vec.size();
	
	if (vec_s == 0) return {nullptr, 0};
	
	auto reserved = vec_s;
	
	for (const auto &str : vec) {
		reserved += str.size();
	}
	
	char *arr = new char [reserved];
	
	size_t pos = 0;
	
	for (size_t i = 0; i < vec_s; ++i) {
		const auto vec_i_size = vec[i].size();
		if (vec_i_size > 0) {
			std::copy(vec[i].data(), vec[i].data() + vec_i_size, arr + pos);
			pos += vec_i_size;
		}
		*(arr + pos) = (i == vec_s - 1) ? '\0' : '\n';
		++pos;
	}
	
	return {arr, reserved};
}

std::vector<std::string> chars_to_str_vec(char *str, size_t n) {
	if (n == 0 || !str) return {};
	
	std::string added_str;
	
	std::vector<std::string> res;
	
	for (size_t i = 0; i < n; ++i) {
		const auto c = str[i];
		if (c == '\n') {
			res.push_back(added_str);
			added_str.clear();
		} else {
			added_str += c;
		}
	}
	
	if (!added_str.empty()) res.push_back(added_str);
	
	return res;
}

static inline size_t get_arr_size(const char *str) {
	size_t i = 0;
	while (*str != '\0') {
		++i;
		++str;
	}
	return i;
}

std::vector<std::string> chars_to_str_vec(char *str) {
	if (!str) return {};
	
	size_t n = get_arr_size(str);
	
	if (n == 0)  return {};
	
	std::string added_str;
	
	std::vector<std::string> res;
	
	for (size_t i = 0; i < n; ++i) {
		const auto c = str[i];
		if (c == '\n') {
			res.push_back(added_str);
			added_str.clear();
		} else {
			added_str += c;
		}
	}
	
	if (!added_str.empty()) res.push_back(added_str);
	
	return res;
}

std::vector<std::string> chars_to_str_vec(const char *str) {
	if (!str) return {};
	
	size_t n = get_arr_size(str);
	
	if (n == 0)  return {};
	
	std::string added_str;
	
	std::vector<std::string> res;
	
	for (size_t i = 0; i < n; ++i) {
		const auto c = str[i];
		if (c == '\n') {
			res.push_back(added_str);
			added_str.clear();
		} else {
			added_str += c;
		}
	}
	
	if (!added_str.empty()) res.push_back(added_str);
	
	return res;
}


double getAccuracy(const char* ac, const char* bc) {
	const auto a = chars_to_str_vec(ac), b = chars_to_str_vec(bc);
	return getAccuracy(a, b);
}

double getAccuracy(const char* ac, const std::vector<std::string>& b) {
	const auto a = chars_to_str_vec(ac);
	
	return getAccuracy(a, b);
}

//referenced
double getBalancedAccuracy(const char* referenced_c, const std::vector<std::string>& obtained) {
	const auto referenced = chars_to_str_vec(referenced_c);
	
	return getBalancedAccuracy(referenced, obtained);
}



