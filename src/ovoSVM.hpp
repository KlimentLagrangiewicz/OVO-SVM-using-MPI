#ifndef OVOSVM_HPP
#define OVOSVM_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

#include "vectors.hpp"
#include "binSVM.hpp"

#ifdef __LP64__
	#ifndef MPI_SIZE_T
		#define MPI_SIZE_T MPI_UNSIGNED_LONG
	#endif
#else
	#ifndef MPI_SIZE_T
		#define MPI_SIZE_T MPI_UNSIGNED
	#endif
#endif

class ovoSVM {
	// Классификаторы
	std::vector<binSVM> binClassifiers;
	
	// Уровни классов и индексы элементов соответ. классов
	std::vector<std::tuple<std::string, std::vector<std::size_t>>> data_vec;
	
	// Тип ядер
	uint8_t k_type = 0;
	
	// описание объектов
	double *x;
	
	// число  объектов
	size_t n;
	
	// Произведения ядер
	double *K;
	
	// число свойств у объектов
	size_t m;
	
	// Параметры полимиального ядра
	double degree, coef0;
	
	// Параметр RBF-ядра
	double gamma;
	
	bool extern_fl = false;
	
	public:
		//Конструктор по умолчанию
		ovoSVM();
		
		//Конструктор
		ovoSVM(std::vector<std::vector<std::string>>& dataX, std::vector<std::string>& dataY);
		
		//
		ovoSVM(double *x, char *str_arr, const size_t n, const size_t m);
		
		//
		ovoSVM(std::vector<std::vector<std::string>>& str_matr);
		
		//
		//ovoSVM(const double *dataX, std::vector<std::string>& dataY);
		
		//Деструктор
		~ovoSVM();
		
		//
		void setC(const double inC);
		
		//
		void setKernelType(const std::string &kT);
		
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
		void setParameters(const std::string &kT = "liner", const double inC = 1.0, const double inB = 0.0, const double inAcc = 0.0001, const int inMaxIt = 10000);
		
		//
		void setData(std::vector<std::vector<double>>& dataX, std::vector<std::string>& dataY);
		
		//
		void kernelCaching();
		
		//
		void MPI_KernelCaching();
		
		//
		void fit();
		
		//
		void MPI_Fit();
		
		//
		std::string predict(const std::vector<double>& _x);
		
		//
		std::string predict(const double* _x);
		
		//
		size_t predictInt(const std::vector<double>& _x);
		
		//
		size_t predictInt(const double* _x);
		
		//
		size_t MPI_PredictInt(const double* _x);
		
		//
		std::string balancedPredict(const std::vector<double>& _x); 
		
		//
		std::vector<std::string> predict(const std::vector<std::vector<double>>& _x);
		
		//
		std::vector<std::string> predict(const double* _x, const size_t n, const size_t m);
		
		//
		std::vector<std::string> MPI_Predict1(const double* _x, const size_t n, const size_t m);
		
		//
		std::vector<std::string> MPI_Predict2(const double* _x, const size_t n, const size_t m);
		
		//
		size_t* predictInt(const double* _x, const size_t begin, const size_t end, const size_t m);
		
		//
		std::vector<std::string> getBalancedPredictions(const std::vector<std::vector<double>>& _x);
		
		//
		ovoSVM& operator = (const ovoSVM& other);
};

#endif