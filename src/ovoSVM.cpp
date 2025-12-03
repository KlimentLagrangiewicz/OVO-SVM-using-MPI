#include "ovoSVM.hpp"


ovoSVM::ovoSVM() {
	
	binClassifiers = {};
	
	k_type = 0;
	
	K = nullptr;
	
	m = 0;
}

template <class T>
static void print(std::vector<T> &vec) {
	for (const auto&v: vec) {
		std::cout << v << '\n';
	}
}

ovoSVM::ovoSVM(double *x, char *str_arr, const size_t n, const size_t m) {
	auto str_vec = chars_to_str_vec(str_arr);
	
	data_vec = get_data_vector(str_vec);
	
	extern_fl = true;
	
	this->x = x;
	this->n = n; 
	this->m = m; 
	
	k_type = 0;
	
	const auto s = data_vec.size();
	if (s < 2) [[unlikely]] throw std::invalid_argument("Unsupported number of classes");
	binClassifiers.resize(s * (s - 1) / 2);
	
	size_t k = 0;
	for (std::size_t i = 0; i < s; ++i) {
		const auto &dv_i = std::get<1>(data_vec[i]);
		
		for (std::size_t j = i + 1; j < s; ++j) {
			const auto &dv_j = std::get<1>(data_vec[j]);
			binClassifiers[k].setData(x, n, m, dv_i.size(), dv_j.size(), dv_i.data(), dv_j.data(), i, j);
			k++;
		}
	}
}


ovoSVM::ovoSVM(std::vector<std::vector<std::string>>& dataX, std::vector<std::string>& dataY) {
	if (dataX.size() != dataY.size()) [[unlikely]] throw std::invalid_argument("X and Y vectors must have the same dimension");
	
	std::tie(x, n, m) = convertToDoubleArray(dataX);
	
	k_type = 0;
	
	data_vec = get_data_vector(dataY);
	const auto s = data_vec.size();
	if (s < 2) [[unlikely]] throw std::invalid_argument("Unsupported number of classes");
	binClassifiers.resize(s * (s - 1) / 2);
	
	size_t k = 0;
	for (std::size_t i = 0; i < s; ++i) {
		const auto &dv_i = std::get<1>(data_vec[i]);
		for (std::size_t j = i + 1; j < s; ++j) {
			const auto &dv_j = std::get<1>(data_vec[j]);
			binClassifiers[k].setData(x, n, m, dv_i.size(), dv_j.size(), dv_i.data(), dv_j.data(), i, j);
			k++;
		}
	}
	
	extern_fl = false;
}


static void print(double *x, size_t n, size_t m) {
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < m; ++j) {
			std::cout << x[i * m + j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

ovoSVM::ovoSVM(std::vector<std::vector<std::string>>& str_matr) {
	if (str_matr.empty()) [[unlikely]] throw std::invalid_argument("Input string matrix is empty");
	
	std::tie(x, n, m) = convertToDoubleArray(str_matr, str_matr[0].size() - 1);
	
	autoscaling(x, n, m);
	
	k_type = 0;
	
	data_vec = get_data_vector(str_matr);
	const auto s = data_vec.size();
	if (s < 2) [[unlikely]] throw std::invalid_argument("Unsupported number of classes");
	binClassifiers.resize(s * (s - 1) / 2);
	
	size_t k = 0;
	for (std::size_t i = 0; i < s; ++i) {
		const auto& dv_i = std::get<1>(data_vec[i]);
		
		for (std::size_t j = i + 1; j < s; ++j) {
			const auto& dv_j = std::get<1>(data_vec[j]);
			binClassifiers[k].setData(x, n, m, dv_i.size(), dv_j.size(), dv_i.data(), dv_j.data(), i, j);
			k++;
			
		}
	}
	
	extern_fl = false;
}

ovoSVM::~ovoSVM() {
	if (!binClassifiers.empty()) binClassifiers.clear();
	if (!data_vec.empty()) data_vec.clear();
	if (K) delete []K;
	K = nullptr;
	
	if (!extern_fl && x) {
		delete []x;
		x = nullptr;
	}
}



void ovoSVM::setC(const double inC) {
	for (auto &a: binClassifiers) {
		a.setC(inC);
	}
}

void ovoSVM::setKernelType(const std::string &kT) {
	if (kT == "rbf") k_type = 1;
	else if (kT == "poly") k_type = 2;
	else k_type = 0;
	for (auto &a: binClassifiers) {
		a.setKernelType(k_type);
	}
}

void ovoSVM::setB(const double inB) {
	for (auto &a: binClassifiers) {
		a.setB(inB);
	}
}

void ovoSVM::setAcc(const double inAcc) {
	for (auto &a: binClassifiers) {
		a.setAcc(inAcc);
	}
}

void ovoSVM::setMaxIt(const int _maxIt) {
	for (auto &a: binClassifiers) {
		a.setMaxIt(_maxIt);
	}
}

void ovoSVM::setGamma(const double gammaIn) {
	gamma = gammaIn;
	for (auto &a: binClassifiers) {
		a.setGamma(gammaIn);
	}
}

void ovoSVM::setPolyParam(const double degreeIn, const double coef0In) {
	degree = degreeIn;
	coef0 = coef0In;
	for (auto &a: binClassifiers) {
		a.setPolyParam(degreeIn, coef0In);
	}
}

void ovoSVM::setParameters(const std::string &kT, const double inC, const double inB, const double inAcc, const int inMaxIt) {
	if (kT == "rbf") k_type = 1;
	else if (kT == "poly") k_type = 2;
	else k_type = 0;
	for (auto &a: binClassifiers) {
		a.setKernelType(k_type);
		a.setC(inC);
		a.setB(inB);
		a.setAcc(inAcc);
		a.setMaxIt(inMaxIt);
	}
}

void ovoSVM::setData(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY) {
	if (dataX.size() != dataY.size()) [[unlikely]] throw std::invalid_argument("X and Y vectors must have the same dimension");
	
	std::tie(x, n, m) = convertToDoubleArray(dataX);

	data_vec = get_data_vector(dataY);
	const auto s = data_vec.size();
	if (s < 2) [[unlikely]] throw std::invalid_argument("Unsupported number of classes");
	binClassifiers.resize(s * (s - 1) / 2);
	size_t k = 0;

	for (std::size_t i = 0; i < s; ++i) {
		const auto& dv_i = std::get<1>(data_vec[i]);
		
		for (std::size_t j = i + 1; j < s; ++j) {
			const auto& dv_j = std::get<1>(data_vec[j]);
			binClassifiers[k].setData(x, n, m, dv_i.size(), dv_j.size(), dv_i.data(), dv_j.data(), i, j);
			k++;
			
		}
	}
}

void ovoSVM::kernelCaching() {
	
	if (k_type == 1) {
		K = gamma == 0.0 ? getKernelMatrix(x, n, m) : getKernelMatrix(x, n, m, gamma);
	} else if (k_type == 2) {
		K = degree == 0.0 ? getKernelMatrix(x, n, m) : getKernelMatrix(x, n, m, coef0, degree);
	} else {
		K = getKernelMatrix(x, n, m);
	}
	
	
	for (auto &a: binClassifiers) {
		a.setKernel(K);
	}
}

void ovoSVM::MPI_KernelCaching() {
	
	if (k_type == 1) {
		
		K = gamma == 0.0 ? MPI_Getkernelmatrix(x, n, m) : MPI_Getkernelmatrix(x, n, m, gamma);
		
	} else if (k_type == 2) {
	
		K = degree == 0.0 ? MPI_Getkernelmatrix(x, n, m) : MPI_Getkernelmatrix(x, n, m, coef0, degree);
		
	} else {
		
		K = MPI_Getkernelmatrix(x, n, m);
		
	}
	
	
	for (auto &a: binClassifiers) {
		a.setKernel(K);
	}
}

void ovoSVM::fit() {
	kernelCaching();
	for (auto &a: binClassifiers) a.fit();
	
	for (auto &a: binClassifiers) a.detAlphas();
}

static inline void sendingDoubleArray(
		double *sendbuf,
		size_t sendsize,
		int sendrank,
		double *recvbuf,
		size_t recvsize,
		int recvrank
	)
{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		
		if (rank == sendrank) {
			//const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm
			MPI_Send(sendbuf, sendsize , MPI_DOUBLE, recvrank, 0, MPI_COMM_WORLD);
		} else if (rank == recvrank) {
			MPI_Status status;
			//void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status
			MPI_Recv(recvbuf, recvsize, MPI_DOUBLE, sendrank, 0, MPI_COMM_WORLD, &status);
		}
}




void ovoSVM::MPI_Fit() {
	MPI_KernelCaching();
	
	int rank, size;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	const auto bin_s = binClassifiers.size();
	
	size_t perProc = bin_s / size;
	if (size == 1) {
		if (rank == 0) for (auto &a: binClassifiers) a.fit();
	} else if (perProc == 0) {
		if (static_cast<size_t>(rank) < bin_s) {
			binClassifiers[rank].fit();
		}
		for (size_t i = 1; i < bin_s; ++i) {
			sendingDoubleArray(binClassifiers[i].getAlphas(), binClassifiers[i].getAlphaSize(), i, binClassifiers[i].getAlphas(), binClassifiers[i].getAlphaSize(), 0);
			
			sendingDoubleArray(binClassifiers[i].getPtrB(), 1, i, binClassifiers[i].getPtrB(), 1, 0);
		}
	} else {
		const size_t begin = rank * perProc;
		const size_t end = begin + perProc;
		for (size_t i = begin; i < end; ++i) {
			binClassifiers[i].fit();
		}
		
		const size_t end2 = size * perProc;
		for (size_t i = perProc; i < end2; ++i) {
			sendingDoubleArray(binClassifiers[i].getAlphas(), binClassifiers[i].getAlphaSize(), i / perProc, binClassifiers[i].getAlphas(), binClassifiers[i].getAlphaSize(), 0);
			
			sendingDoubleArray(binClassifiers[i].getPtrB(), 1, i / perProc, binClassifiers[i].getPtrB(), 1, 0);
		}
		
		const auto diff = bin_s - end2;
		
		if (diff > 0) {
			if (static_cast<size_t>(rank) < diff) {
				binClassifiers[end2 + rank].fit();
			}
			
			for (size_t i = 1; i < diff; ++i) {
				sendingDoubleArray(binClassifiers[end2 + i].getAlphas(), binClassifiers[end2 + i].getAlphaSize(), i, binClassifiers[end2 + i].getAlphas(), binClassifiers[end2 + i].getAlphaSize(), 0);
				
				sendingDoubleArray(binClassifiers[end2 + i].getPtrB(), 1, i, binClassifiers[end2 + i].getPtrB(), 1, 0);
			}
		}
	}
	
	for (auto &a: binClassifiers) {
		MPI_Bcast(a.getAlphas(), a.getAlphaSize(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(a.getPtrB(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
	}
	
	for (auto &a: binClassifiers) a.detAlphas();
}

std::string ovoSVM::predict(const std::vector<double>& _x) {
	if (binClassifiers.empty()) return "";
	std::vector<std::size_t> vec(data_vec.size(), 0);
	for (const auto &a: binClassifiers) {
		const auto p = a.predict(_x.data());
		++vec[p];
	}
	const auto max_it = std::max_element(vec.begin(), vec.end());
	if (max_it != vec.end()) {
		const std::size_t max_index = std::distance(vec.begin(), max_it);
		const auto& str = std::get<0>(data_vec[max_index]);
		//std::cout << "str: " << str << '\n';
		return str;
	}
	return "";

}


std::string ovoSVM::predict(const double* _x) {
	if (binClassifiers.empty()) return "";
	std::vector<std::size_t> vec(data_vec.size(), 0);
	for (const auto &a: binClassifiers) {
		const auto p = a.predict(_x);
		++vec[p];
	}
	const auto max_it = std::max_element(vec.begin(), vec.end());
	if (max_it != vec.end()) {
		const std::size_t max_index = std::distance(vec.begin(), max_it);
		const auto& str = std::get<0>(data_vec[max_index]);
		//std::cout << "str: " << str << '\n';
		return str;
	}
	return "";

}

size_t ovoSVM::predictInt(const std::vector<double>& _x) {
	if (binClassifiers.empty()) return 0;
	std::vector<std::size_t> vec(data_vec.size(), 0);
	for (const auto &a: binClassifiers) {
		const size_t _num = a.predict(_x.data());
		++vec[_num];
	}
	const auto max_it = std::max_element(vec.begin(), vec.end());
	
	if (max_it != vec.end()) return std::distance(vec.begin(), max_it);
	
	return 0;
}


size_t ovoSVM::predictInt(const double* _x) {
	if (binClassifiers.empty() || !_x) return 0;
	std::vector<std::size_t> vec(data_vec.size(), 0);
	for (const auto &a: binClassifiers) {
		const size_t _num = a.predict(_x);
		++vec[_num];
	}
	const auto max_it = std::max_element(vec.begin(), vec.end());
	
	if (max_it != vec.end()) return std::distance(vec.begin(), max_it);

	return 0;
}

size_t ovoSVM::MPI_PredictInt(const double* _x) {
	if (binClassifiers.empty() || !_x) return 0;
	
	size_t res = 0;
	
	int size, rank;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	const auto bin_size = binClassifiers.size();
	
	size_t perProc = bin_size / size;
	
	if (size == 1) {
		if (rank == 0) res = predictInt(_x);
		
		//MPI_Bcast(&res, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
		
	} else if (perProc == 0) {
		const auto dv_size = data_vec.size();
	
		size_t *vec = new size_t [dv_size];
		std::fill(vec, vec + dv_size, 0);
		
		if (static_cast<size_t>(rank) < bin_size) {
			const auto _num = binClassifiers[rank].predict(_x);
			++vec[_num];
		}
		
		if (rank == 0) MPI_Reduce(MPI_IN_PLACE, vec, dv_size, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
		else MPI_Reduce(vec, vec, dv_size, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
		
		if (rank == 0) {
			const auto max_it = std::max_element(vec, vec + dv_size);
			res = max_it == vec + dv_size ? 0 : std::distance(vec, max_it);
		}
		
		delete []vec;
		
		//MPI_Bcast(&res, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
	} else {
		const auto dv_size = data_vec.size();
	
		size_t *vec = new size_t [dv_size];
		std::fill(vec, vec + dv_size, 0);
		
		
		const auto begin = rank * perProc;
		const auto end = begin + perProc;
		for (size_t i = begin; i < end; ++i) {
			const auto _num = binClassifiers[i].predict(_x);
			++vec[_num];
		}
		
		const auto buff = size * perProc;
		if (bin_size > buff && rank == 0) {
			for (size_t i = buff; i < bin_size; ++i) {
				const auto _num = binClassifiers[i].predict(_x);
				++vec[_num];
			}
		}
		
		if (rank == 0) MPI_Reduce(MPI_IN_PLACE, vec, dv_size, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
		else MPI_Reduce(vec, vec, dv_size, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
		
		
		if (rank == 0) {
			const auto max_it = std::max_element(vec, vec + dv_size);
			res = max_it == vec + dv_size ? 0 : std::distance(vec, max_it);
		}
		delete []vec;
		
	}
	
	MPI_Bcast(&res, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
	
	/*
	if (perProc == 0 || size == 1) {
		if (rank == 0) res = predictInt(_x);
		
		MPI_Bcast(&res, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
		
	} else {
		const auto dv_size = data_vec.size();
	
		size_t *vec = new size_t [dv_size];
		std::fill(vec, vec + dv_size, 0);
		
		
		const auto begin = rank * perProc;
		const auto end = begin + perProc;
		for (size_t i = begin; i < end; ++i) {
			const auto _num = binClassifiers[i].predict(_x);
			++vec[_num];
		}
		
		const auto buff = size * perProc;
		if (bin_size > buff && rank == 0) {
			for (size_t i = buff; i < bin_size; ++i) {
				const auto _num = binClassifiers[i].predict(_x);
				++vec[_num];
			}
		}
		
		if (rank == 0) MPI_Reduce(MPI_IN_PLACE, vec, dv_size, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
		else MPI_Reduce(vec, vec, dv_size, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
		
		
		if (rank == 0) {
			const auto max_it = std::max_element(vec, vec + dv_size);
			res = max_it == vec + dv_size ? 0 : std::distance(vec, max_it);
		}
		delete []vec;
		MPI_Bcast(&res, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
	}
	*/
	
	return res;
}

std::string ovoSVM::balancedPredict(const std::vector<double>& _x) {
	if (binClassifiers.empty()) return "";
	std::vector<double> vec(data_vec.size(), 0.0);
	for (auto &a: binClassifiers) {
		const auto i = a.getPredictPair(_x.data());
		vec[i.first] += i.second;
	}
	const auto max_it = std::max_element(vec.begin(), vec.end());
	if (max_it != vec.end()) {
		const std::size_t max_index = std::distance(vec.begin(), max_it);
		return std::get<0>(data_vec[max_index]);
	}
	return "";
}

std::vector<std::string> ovoSVM::predict(const std::vector<std::vector<double>>& _x) {
	std::vector<std::string> res;
	res.reserve(_x.size());
	for (const auto &xi: _x) {
		res.push_back(predict(xi));
	}
	return res;
}

std::vector<std::string> ovoSVM::predict(const double* _x, const size_t n, const size_t m) {
	std::vector<std::string> res;
	res.reserve(n);
	size_t offset = 0;
	for (size_t i = 0; i < n; ++i) {
		res.push_back(predict(_x + offset));
		offset += m;
	}
	
	return res;
}


std::vector<std::string> ovoSVM::MPI_Predict1(const double* _x, const size_t n, const size_t m) {
	size_t *res_int;
	
	int size, rank;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	size_t perProc = n / size;
	
	if (perProc == 0 || size == 1) {
		if (rank == 0) res_int = predictInt(_x, 0, n, m);
		else res_int = new size_t [n];
		// res_int = rank == 0 ? predictInt(_x, 0, n, m) : new size_t [n];
		MPI_Bcast(res_int, n, MPI_SIZE_T, 0, MPI_COMM_WORLD);
	} else {
		res_int = new size_t [n];
		
		const size_t begin = rank * perProc;
		const size_t end = begin + perProc;
		
		size_t offset = begin * m;
		
		for (size_t i = begin; i < end; ++i) {
			res_int[i] = predictInt(_x + offset);
			offset += m;
		}
		
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, res_int, perProc, MPI_SIZE_T, MPI_COMM_WORLD);
		
		auto buffer = perProc * size;
		if (n > buffer) {
			if (rank == 0) {
				offset = buffer * m;
				for (size_t i = buffer; i < n; ++i) {
					res_int[i] = predictInt(_x + offset);
					offset += m;
				}
			}
			MPI_Bcast(res_int + buffer, n - buffer, MPI_SIZE_T, 0, MPI_COMM_WORLD);
		}
	}
	
	std::vector<std::string> res(n);
	
	for (size_t i = 0; i < n; ++i) {
		res[i] = std::get<0>(data_vec[res_int[i]]);
	}
	
	delete []res_int;
	
	return res;
}

std::vector<std::string> ovoSVM::MPI_Predict2(const double* _x, const size_t n, const size_t m) {
	std::vector<size_t> res_size(n);
	size_t offset = 0;
	for (size_t i = 0; i < n; ++i) {
		res_size[i] = MPI_PredictInt(_x + offset);
		offset += m;
	}
	
	std::vector<std::string> res(n);
	for (size_t i = 0; i < n; ++i) {
		res[i] = std::get<0>(data_vec[res_size[i]]);
	}
	
	return res;
}


size_t* ovoSVM::predictInt(const double* _x, const size_t begin, const size_t end, const size_t m) {
	size_t *res = new size_t [end - begin];
	
	size_t offset = begin * m;
	for (size_t i = begin; i < end; ++i) {
		res[i - begin] = predictInt(_x + offset);
		offset += m;
	}
	
	return res;
}

std::vector<std::string> ovoSVM::getBalancedPredictions(const std::vector<std::vector<double>>& _x) {
	std::vector<std::string> res;
	res.reserve(_x.size());
	for (const auto &xi: _x) {
		res.push_back(balancedPredict(xi));
	}
	return res;
}

ovoSVM& ovoSVM::operator = (const ovoSVM& other) {
	if (!other.binClassifiers.empty()) binClassifiers = other.binClassifiers;
	if (!other.data_vec.empty()) data_vec = other.data_vec;
	if (other.K) K = other.K;
	if (other.x) x = other.x;
	n = other.n;
	m = other.m;
	
	k_type = other.k_type;
	
	if (k_type == 2){
		degree = other.degree;
		coef0 = other.coef0;
	}
	if (k_type == 1) gamma = other.gamma;
	
	return *this;
}
