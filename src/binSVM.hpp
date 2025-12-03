#ifndef BINSVM_HPP
#define BINSVM_HPP


#include "vectors.hpp"

class binSVM {
	// Тип ядра: 0 - линейное, 1 - rbf, 2 - poly
	uint8_t kernelType; 
	
	// Матрица ядер
	const double *K;
	
	// Матрица с свойства обучающей выборки
	const double *x;
	
	// Общее число объектов и число их свойств
	std::size_t n, m;
	
	// номера объектов для обучения с метками "+" (p, positive) и "-" (n, negative)
	const std::size_t *idp, *idn;
	
	// номера объектов с активными альфа-папрметрами
	std::size_t *ida;
	
	// количество объектов с метками "+" и "-"
	std::size_t np, nn, np_a, nn_a;
	
	// Параметры опорных векторов
	double *a;
	
	// Параметр регуляризации
	double c;
	
	// Порог для SVM 
	double b;
	
	// Пороговая точность
	double acc;
	
	// Максимальное количество итераций
	std::size_t maxIt;
	
	// Уровни меток
	std::size_t labels_first, labels_second;
	
	// Параметры полимиального ядра
	double degree, coef0;
	
	// Параметр RBF-ядра
	double gamma;
	
	// Проверка нарушения условий Каруша — Куна — Таккера
	inline bool check_kkt(const size_t check_idx, const double fxi) const;
	
	//
	inline std::pair<bool, double> get_violation(const double fxi, const size_t i) const;
	
	//
	double get_violation_value(const double fxi, const size_t i) const;
	
	//
	size_t select_i(const double *fx) const;
	
	//
	std::pair<size_t, size_t> select_working_set(const double* const fx);
	
	//
	inline double get_error(const size_t i, const double fxi) const;
	
	//
	double get_delta(const double* const fx, const size_t i, const size_t j) const;
	
	//
	double get_delta(const double* fx, const size_t i, const size_t j, const double y_i, const double ai, const size_t i_glob, const double k_ii, const double Ei) const;
	
	//
	void update_fx(double *vec, const size_t i, const size_t j, const double di, const double dj, const double db) const;
	
	//
	void calcSMO(const double* const fx, const size_t i, const size_t j);
	
	//
	double getKerlenPr(const double* const _x, const size_t number) const;
	
	//
	double finalF(const double* const _x) const;
	
	
	public:
		// конструктор
		binSVM(const uint8_t kT = 0, const double inC = 1.0, const double inB = 0.0, const double inAcc = 0.0001, const int inMaxIt = 10000, size_t inM = 0);
		
		// деструктор
		~binSVM();
		
		//
		void fit();
		
		//
		int predictInt(const double* const _x) const;
		
		//
		std::size_t predict(const double* const _x) const;
		
		//
		std::pair<std::size_t, double> getPredictPair(const double* const _x) const;
		
		//
		int getNumOfAttribytes();
		
		//
		void setData(const double* in_x, const double* in_k, const size_t _n, const size_t _m, const std::size_t np_s, const std::size_t nn_s, const size_t *_idp, const size_t *_idn);
		
		//
		void setData(const double* in_x, const size_t _n, const size_t _m, const std::size_t np_s, const std::size_t nn_s, const size_t *_idp, const size_t *_idn);
		
		//
		void setData(const double* in_x, const size_t _n, const size_t _m, const std::size_t np_s, const std::size_t nn_s, const size_t *_idp, const size_t *_idn, const std::size_t l_p, const std::size_t l_n);
		
		//
		void setKernel(const double* in_k, const size_t _n);
		
		//
		void setKernel(const double* in_k);
		
		//
		void setC(const double inC);
		
		//
		void setKernelType(const uint8_t kT);
		
		//
		void setB(const double inB);
		
		//
		void setAcc(const double inAcc);
		
		//
		void setMaxIt(const int _maxIt);
		
		//
		void setGamma(const double gammaIn = 0.0);
		
		//
		void setPolyParam(const double degreeIn = 0.0, const double coef0In = 0.0);
		
		//
		void setParameters(const uint8_t kT = 0, const double inC = 1.0, const double inB = 0.0, const double inAcc = 0.0001, const int inMaxIt = 10000);
		
		//
		binSVM& operator = (const binSVM& other);
		
		//
		void detAlphas();
		
		//
		size_t getAlphaSize();
		
		//
		double* getAlphas();
		
		//
		void setAlphas(double *a);
		
		double* getPtrB();
		
};

#endif