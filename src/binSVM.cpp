#include "binSVM.hpp"


binSVM::binSVM(const uint8_t kT, const double inC, const double inB, const double inAcc, const int inMaxIt, size_t inM) {
	kernelType = (kT != 0 && kT != 1 && kT != 2) ? 0 : kT;
	c = inC;
	b = inB;
	acc = inAcc <= 0.0 ? 0.0001 : inAcc;
	maxIt = inMaxIt < 1 ? 10000 : inMaxIt;
	degree = 0.0;
	coef0 = 0.0;
	gamma = 0.0;
	x = nullptr;
	K = nullptr;
	a = nullptr;
	ida = nullptr;
	m = inM;
}

binSVM::~binSVM() {
	if (a) delete []a;
	a = nullptr;
}

bool binSVM::check_kkt(const size_t i, const double fxi) const {
	bool flag = false;
	const double y_i = i < np ? 1.0 : -1.0;
	if (a[i] < acc) flag = y_i * fxi >= 1.0 - acc;
	else if (a[i] > c - acc) flag = y_i * fxi <= 1.0 + acc;
	else flag = std::abs(y_i * fxi - 1.0) <= acc;
	return flag;
}

inline std::pair<bool, double> binSVM::get_violation(const double fxi, const size_t i) const {
	const double ai    = a[i];
	const double y_fx  = (i < np) ? fxi : -fxi;
	const double g     = y_fx - 1.0;
	bool bad;
	double v;
	
	if (ai <= acc) {
		bad = g < -acc;
		v   = std::max(0.0, -g);
	} else if (ai >= c - acc) {
		bad = g > acc;
		v   = std::max(0.0,  g);
	} else {
		bad = std::fabs(g) > acc;
		v   = std::fabs(g);
	}
	
	return std::make_pair(bad, v);
}

double binSVM::get_violation_value(const double fxi, const size_t i) const {
	const double y_i = i < np ? 1.0 : -1.0;	
	/*if (a[i] < acc) {
		return std::max(0.0, 1.0 - y_i * fxi);
	} else if (a[i] > c - acc) {
		return std::max(0.0, y_i * fxi - 1.0);
	}*/
	return std::abs(fxi * y_i - 1.0);
}

size_t binSVM::select_i(const double *fx) const {
	double max_violation = 0.0;
	
	const size_t total = np + nn;
	
	size_t i = total;
	
	for (size_t k = 0; k < total; ++k) {
		if (const auto [bad, viol] = get_violation(fx[k], k); bad && viol > max_violation) {
			max_violation = viol;
			i = k;
		}
	}
	
	if (max_violation <= acc) {
		return total;
	}
	
	return i;
}

std::pair<size_t, size_t> binSVM::select_working_set(const double* const fx) {
	size_t i = select_i(fx);
	
	const size_t total = np + nn;
	
	if (i == total) return std::make_pair(total, total);
	
	const double y_i = i < np ? 1.0 : -1.0;
	const double ai = a[i];
	
	const size_t i_glob = i < np ? idp[i] : idn[i - np];
	const double k_ii = K[i_glob * n + i_glob];
	const double Ei = get_error(i, fx[i]);
	
	double max_gain = 0.0;
	
	size_t j = total;
	
	for (size_t k = 0; k < total; ++k) {
		if (const double cur_gain = get_delta(fx, i, k, y_i, ai, i_glob, k_ii, Ei); cur_gain > max_gain) {
			max_gain = cur_gain;
			j = k;
		}
	}
	
	if (j == total) return std::make_pair(total, total);
	
	return std::make_pair(i, j);
}

inline double binSVM::get_error(const size_t i, const double fxi) const {
	return fxi - (i < np ? 1 : -1);
}

double binSVM::get_delta(const double* const fx, const size_t i, const size_t j) const {
	if (i == j) return 0.0;
	
	const double y_i = i < np ? 1.0 : -1.0;	
	const double y_j = j < np ? 1.0 : -1.0;	
	
	const double ai = a[i], aj = a[j];
	
	double L, H;
	if (y_i == y_j) {
		const double sum  = ai + aj;
		L = sum - c;
		H = sum;
	} else {
		const double diff = aj - ai;
		L = diff;
		H = c + diff;
	}
	
	if (L < 0.0) L = 0.0;
	if (H > c)   H = c;
	
	if (L == H) return 0.0;
	
	const size_t i_glob = i < np ? idp[i] : idn[i - np], j_glob = j < np ? idp[j] : idn[j - np];
	const double k_ij = K[i_glob * n + j_glob], k_ii = K[i_glob * n + i_glob], k_jj = K[j_glob * n + j_glob];
	const double nu = 2.0 * k_ij - k_ii - k_jj;
	
	if (nu >= 0.0) return 0.0;
	const double Ei = get_error(i, fx[i]), Ej = get_error(j, fx[j]);
	
	double a_j_new = aj - y_j * (Ei - Ej) / nu;
	
	if (a_j_new > H) a_j_new = H;
	if (a_j_new < L) a_j_new = L;
	
	
	const double delta = a_j_new - aj;
	
	if (std::abs(delta) < acc) return 0.0;
	
	//return delta < acc ? 0.0 : delta;
	
	return y_j *(Ei - Ej) * delta + 0.5 * nu * delta * delta;
}

double binSVM::get_delta(const double* fx, const size_t i, const size_t j, const double y_i, const double ai, const size_t i_glob, const double k_ii, const double Ei) const {
	if (i == j) return 0.0;
	
	const double y_j = j < np ? 1.0 : -1.0;	
	
	const double aj = a[j];
	
	double L, H;
	if (y_i == y_j) {
		const double sum  = ai + aj;
		L = sum - c;
		H = sum;
	} else {
		const double diff = aj - ai;
		L = diff;
		H = c + diff;
	}
	
	if (L < 0.0) L = 0.0;
	if (H > c)   H = c;
	
	if (L == H) return 0.0;
	
	const size_t j_glob = j < np ? idp[j] : idn[j - np];
	const double k_ij = K[i_glob * n + j_glob], k_jj = K[j_glob * n + j_glob];
	const double nu = 2.0 * k_ij - k_ii - k_jj;
	
	if (nu >= 0.0) return 0.0;
	const double Ej = get_error(j, fx[j]);
	
	double a_j_new = aj - y_j * (Ei - Ej) / nu;
	
	if (a_j_new > H) a_j_new = H;
	if (a_j_new < L) a_j_new = L;
	
	
	const double delta = a_j_new - aj;
	
	if (std::abs(delta) < acc) return 0.0;
	
	//return delta < acc ? 0.0 : delta;
	
	//return y_j *(Ei - Ej) * delta + 0.5 * nu * delta * delta;
	return std::max(0.0, y_j * (Ei - Ej) * delta + 0.5 * nu * delta * delta);
}

void binSVM::update_fx(double *vec, const size_t i, const size_t j, const double di, const double dj, const double db) const {
	const double y_i = i < np ? di : -di;
	const double y_j = j < np ? dj : -dj;
	
	const double* const K_i = K + (i < np ? idp[i] : idn[i - np]) * n;
	const double* const K_j = K + (j < np ? idp[j] : idn[j - np]) * n;
	
	for (size_t l = 0; l < np; ++l) {
		vec[l] += db + y_i * K_i[idp[l]] + y_j * K_j[idp[l]];
	}
	
	for (size_t l = np; l < np + nn; ++l) {
		vec[l] += db + y_i * K_i[idn[l - np]] + y_j * K_j[idn[l - np]];
	}
}


template <class T>
static void my_memset(T* arr, size_t s, T val = (T)0) {
	if (!arr || s == 0u) return;
	
	std::fill(
		std::execution::unseq,
		arr,
		arr + s,
		val
	);
}

void binSVM::calcSMO(const double* const fx, const size_t i, const size_t j) {
	const double y_i = i < np ? 1.0 : -1.0;	
	const double y_j = j < np ? 1.0 : -1.0;	
	
	const double Ei = get_error(i, fx[i]), Ej = get_error(j, fx[j]);
	
	const double ai = a[i], aj = a[j];
	
	double L, H;
	if (y_i == y_j) {
		const double sum  = ai + aj;
		L = sum - c;
		H = sum;
	} else {
		const double diff = aj - ai;
		L = diff;
		H = c + diff;
	}
	
	if (L < 0.0) L = 0.0;
	if (H > c)   H = c;
	
	const size_t i_glob = i < np ? idp[i] : idn[i - np], j_glob = j < np ? idp[j] : idn[j - np];
	const double k_ij = K[i_glob * n + j_glob], k_ii = K[i_glob * n + i_glob], k_jj = K[j_glob * n + j_glob];
	
	const double nu = 2.0 * k_ij - k_ii - k_jj;
	
	a[j] -= y_j * (Ei - Ej) / nu;
	if (a[j] > H) a[j] = H;
	if (a[j] < L) a[j] = L;
	const double delta_j = a[j] - aj;
	const double delta_i = -y_i * y_j * delta_j;
	a[i] = ai + delta_i;
	
	const double b1 = b - Ei - y_i * delta_i * k_ii - y_j * delta_j * k_ij;
	const double b2 = b - Ej - y_i * delta_i * k_ij - y_j * delta_j * k_jj;
	
	if (0.0 < a[i] && a[i] < c) b = b1;
	else if (0.0 < a[j] && a[j] < c) b = b2;
	else b = (b1 + b2) / 2.0;
}


void binSVM::fit() {
	np_a = 0;
	nn_a = 0;
	
	const auto total = np + nn;
	
	double *fx = new double [total];
	my_memset<double>(a, total, 0.0);
	my_memset<double>(fx, total, 0.0);
	bool flag = true;
	for (size_t count = 0; count < maxIt && flag; ++count) {
		const auto &[i, j] = select_working_set(fx);
		if (i == total) flag = false;
		else {
			const double ai_old = a[i], aj_old = a[j], b_old = b;
			calcSMO(fx, i, j);
			update_fx(fx, i, j, a[i] - ai_old, a[j] - aj_old, b - b_old);
		}
	}
	
	delete []fx;
}

/*
double binSVM::getKerlenPr(const std::vector<double>& _x, const size_t number) const {
	if (kernelType == 1) {
		if (gamma == 0.0) return scalar_product(x[number], _x);
		else {
			const double d = euclidean_distance(x[number], _x);
			return std::exp(-gamma * d * d);
		}
	} else if (kernelType == 2) {
		if (degree == 0.0) return scalar_product(x[number], _x);
		else {
			const double s = scalar_product(x[number], _x);
			return std::pow(s + coef0, degree);
		}
	} else {
		return scalar_product(x[number], _x);
	}
	return 0.0;
}
*/

double binSVM::getKerlenPr(const double* const _x, const size_t number) const {
	if (kernelType == 1) {
		if (gamma == 0.0) return scalar_product(x + number * m, _x, m);
		else {
			const double d = euclidean_distance(x + number * m, _x, m);
			return std::exp(-gamma * d * d);
		}
	} else if (kernelType == 2) {
		if (degree == 0.0) return scalar_product(x + number * m, _x, m);
		else {
			const double s = scalar_product(x + number * m, _x, m);
			return std::pow(s + coef0, degree);
		}
	} else {
		return scalar_product(x + number * m, _x, m);
	}
}

/*
double binSVM::finalF(const std::vector<double>& _x) const {
	double res = 0.0;
	for (const auto e: nums) {
		res += a[e] * y[e] * getKerlenPr(_x, e);
	}
	return res + b;
}
*/

double binSVM::finalF(const double* const _x) const {
	double res = 0.0;
	if (!ida || (np_a == 0 && nn_a == 0)) {
		for (size_t i = 0; i < np; ++i) {
			if (std::abs(a[i]) > acc) res += a[i] * getKerlenPr(_x, idp[i]);
		}
		
		for (size_t i = np; i < np + nn; ++i) {
			if (std::abs(a[i]) > acc) res -= a[i] * getKerlenPr(_x, idn[i - np]);
		}
	} else {
		for (size_t i = 0; i < np_a; ++i) {
			const auto id = ida[i];
			res += a[id] * getKerlenPr(_x, idp[id]);
		}
		
		for (size_t i = np_a; i < np_a + nn_a; ++i) {
			const auto id = ida[i];
			res -= a[id] * getKerlenPr(_x, idn[id - np]);
		}
	}
	return res + b;
}


int binSVM::predictInt(const double* const _x) const {
	const double v = finalF(_x);
	if (v >= 0.0) return 1;
	else return -1;
}


std::size_t binSVM::predict(const double* const _x) const {
	const double v = finalF(_x);
	return (v >= 0.0) ? labels_first : labels_second;
}

std::pair<std::size_t, double> binSVM::getPredictPair(const double* const _x) const {
	const double v = finalF(_x);
	const auto key = (v >= 0.0) ? labels_first : labels_second;
	const double w = 1.0 / (1.0 + std::exp(-std::abs(v)));
	return std::make_pair(key, w);
}


int binSVM::getNumOfAttribytes() {
	return m;
}

void binSVM::setData(const double* in_x, const double* in_k, const size_t _n, const size_t _m, const std::size_t np_s, const std::size_t nn_s, const size_t *_idp, const size_t *_idn) {
	x = in_x;
	K = in_k;
	np = np_s;
	nn = nn_s;
	idp = _idp;
	idn = _idn;
	n = _n;
	m = _m;
	
	if (a) delete []a;
	a = new double [np + nn];
}

void binSVM::setData(const double* in_x, const size_t _n, const size_t _m, const std::size_t np_s, const std::size_t nn_s, const size_t *_idp, const size_t *_idn) {
	x = in_x;
	np = np_s;
	nn = nn_s;
	idp = _idp;
	idn = _idn;
	n = _n;
	m = _m;
	
	if (a) delete []a;
	a = new double [np + nn];
}

void binSVM::setData(const double* in_x, const size_t _n, const size_t _m, const std::size_t np_s, const std::size_t nn_s, const size_t *_idp, const size_t *_idn, const std::size_t l_p, const std::size_t l_n) {
	x = in_x;
	np = np_s;
	nn = nn_s;
	idp = _idp;
	idn = _idn;
	n = _n;
	m = _m;
	labels_first = l_p;
	labels_second = l_n;
	
	if (a) delete []a;
	a = new double [np + nn];
}


void binSVM::setKernel(const double* in_k, const size_t _n) {
	K = in_k;
	n = _n;
}

void binSVM::setKernel(const double* in_k) {
	K = in_k;
}

void binSVM::setC(const double inC) {
	c = inC;
}

void binSVM::setKernelType(const uint8_t kT) {
	kernelType = (kT != 0 && kT != 1 && kT != 2) ? 0 : kT;
}

void binSVM::setB(const double inB) {
	b = inB;
}

void binSVM::setAcc(const double inAcc) {
	acc = inAcc;
}

void binSVM::setMaxIt(const int _maxIt) {
	maxIt = _maxIt;
}

void binSVM::setGamma(const double gammaIn) {
	gamma = gammaIn;
}

void binSVM::setPolyParam(const double degreeIn, const double coef0In) {
	degree = degreeIn;
	coef0 = coef0In;
}

void binSVM::setParameters(const uint8_t kT, const double inC, const double inB, const double inAcc, const int inMaxIt) {
	kernelType = (kT != 0 && kT != 1 && kT != 2) ? 0 : kT;
	c = inC;
	b = inB;
	acc = inAcc <= 0.0 ? 0.0001 : inAcc;
	maxIt = inMaxIt < 1 ? 10000 : inMaxIt;	
	degree = 0.0;	
	coef0 = 0.0;	
	gamma = 0.0;
}

template <class T>
static T* my_memcopy(T* arr, size_t s) {
	if (!arr || s == 0u) return nullptr;
	
	T *res = new T[s];
	if (res == nullptr) return nullptr;
	
	T *ptr = res;
	for (; s > 0; --s) {
		*ptr = *arr;
		++ptr;
		++arr;
	}
	
	return res;
}

binSVM& binSVM::operator = (const binSVM& other) {
	
	np = other.np;
	nn = other.nn;
	if (other.a) {
		a = my_memcopy(other.a, other.np + other.nn);
	}
	
	np_a = other.np_a;
	nn_a = other.nn_a;
	
	if (other.ida) {
		ida = my_memcopy(other.ida, other.np_a + other.nn_a);
	}
	
	
	K = other.K;
	x = other.x;
		
	c = other.c;
	b = other.b;
	acc = other.acc;
	maxIt = other.maxIt;
	labels_first = other.labels_first;
	labels_second = other.labels_second;
	
	kernelType = (other.kernelType != 0 && other.kernelType != 1 && other.kernelType != 2) ? 0 : other.kernelType;
	degree = other.degree;
	coef0 = other.coef0;
	gamma = other.gamma;
	
	return *this;
}

void binSVM::detAlphas() {
	np_a = 0;
	nn_a = 0;
	for (size_t i = 0; i < np; ++i) {
		if (std::abs(a[i]) > acc) ++np_a;
	}
	
	for (size_t i = np; i < np + nn; ++i) {
		if (std::abs(a[i]) > acc) ++nn_a;
	}
	
	ida = new size_t [np_a + nn_a];
	size_t k = 0;
	for (size_t i = 0; i < np + nn; ++i) {
		if (std::abs(a[i]) > acc) {
			ida[k] = i;
			++k;
		}
	}
}

size_t binSVM::getAlphaSize() {
	return np + nn;
}

double* binSVM::getAlphas() {
	return a;
}


void binSVM::setAlphas(double *a) {
	this->a = a;
}

double* binSVM::getPtrB() {
	return &(this->b);
}
