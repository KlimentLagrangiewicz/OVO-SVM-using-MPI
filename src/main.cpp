#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <mpi.h>

#include "ovoSVM.hpp"
#include "vectors.hpp"


#ifdef __LP64__
	#ifndef MPI_SIZE_T
		#define MPI_SIZE_T MPI_UNSIGNED_LONG
	#endif
#else
	#ifndef MPI_SIZE_T
		#define MPI_SIZE_T MPI_UNSIGNED
	#endif
#endif

#ifndef ACCURACY_MESURE
	#define ACCURACY_MESURE 200
#endif


template <class T>
static inline void deleting(T* &arr) {
	if (arr) delete []arr;
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	
	try {
		if (argc < 3) {
			std::cout << "Not enough parameters!\n";
		} else {
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			
			int size;
			MPI_Comm_size(MPI_COMM_WORLD, &size);
			
			bool parallel = size > 1;
			
			double *x_train = nullptr, *x_test = nullptr;
			char *y_train = nullptr, *y_test = nullptr;
			
			size_t n_train = 0, m_train = 0, y_train_s = 0;
			
			size_t n_test = 0, m_test = 0, y_test_s = 0;
			
			bool flag1, flag2;
			if (rank == 0) {
				
				std::string trainfile(argv[1]), testfile(argv[2]);
				
				flag2 = trainfile != testfile;
				
				std::vector<std::vector<std::string>> strTrainData = readCSV(trainfile);
				if (!all_equal_size(strTrainData)) throw std::runtime_error("All vectors must have equal size\n");
				
				std::vector<std::vector<std::string>> strTestData = readCSV(testfile);
				if (!all_equal_size(strTestData)) throw std::runtime_error("All vectors must have equal size\n");
				
				trimStringMatrixSpaces(strTrainData);
				trimStringMatrixSpaces(strTestData);
				
				flag1 = strTrainData[0].size() == strTestData[0].size();
				
				if (!validateAllButLastAsDouble(strTrainData[0])) {
					removeFirstElement(strTrainData);
				}
				
				if (!validateAllButLastAsDouble(strTestData[0])) {
					removeFirstElement(strTestData);
				}
				
				std::tie(x_train, n_train, m_train) = convertToDoubleArray(strTrainData, strTrainData[0].size() - 1);
				
				autoscaling(x_train, n_train, m_train);
				
				const auto y_str_train = getLastTable(strTrainData);
				
				std::tie(y_train, y_train_s) = vec_to_chars(y_str_train);
				
				if (flag1) {
					std::tie(x_test, n_test, m_test) = convertToDoubleArray(strTestData, strTestData[0].size() - 1);
					
					const auto y_str_test = getLastTable(strTestData);
					
					std::tie(y_test, y_test_s) = vec_to_chars(y_str_test);					
				} else {
					std::tie(x_test, n_test, m_test) = convertToDoubleArray(strTestData, strTestData[0].size());
				}
				
			}
			
			MPI_Bcast(&flag1, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
			
			MPI_Bcast(&flag2, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
			
			MPI_Bcast(&n_train, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
			
			MPI_Bcast(&m_train, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
			
			MPI_Bcast(&n_test, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
			
			MPI_Bcast(&m_test, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
			
			MPI_Bcast(&y_train_s, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
			
			if (flag1)
				
				MPI_Bcast(&y_test_s, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
			
			if (rank != 0) {
				
				x_train = new double [n_train * m_train];
				
				y_train = new char [y_train_s];
				
				
				x_test = new double [n_test * m_test];
				
				if (flag1) {
					y_test = new char [y_test_s];
				}
			}
			
			MPI_Bcast(x_train, n_train * m_train, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
			MPI_Bcast(y_train, y_train_s, MPI_CHAR, 0, MPI_COMM_WORLD);
			
			MPI_Bcast(x_test, n_test * m_test, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			
			if (flag1) {
				MPI_Bcast(y_test, y_test_s, MPI_CHAR, 0, MPI_COMM_WORLD);
			}
			
			if (flag1) { // Если размерности (число свойств + 1) данных совпадают, значит, вместе с тестовой выборкой поданы и их метки
				const double c = (argc > 4) ? std::stod(argv[4]) : 1.0;
				const double b = (argc > 5) ? std::stod(argv[5]) : 0.0;
				const double acc = (argc > 6) ? std::stod(argv[6]) : 0.0001;
				const int maxIt = (argc > 7) ? std::stoi(argv[7]) : 10000;
				std::string kernelType = (argc > 8) ? std::string(argv[8]) : "liner";
				
				ovoSVM ovo_svm(x_train, y_train, n_train, m_train);
				
				ovo_svm.setParameters(kernelType, c, b, acc, maxIt);
				
				if (kernelType == "rbf") {
					const double gamma = (argc > 9) ? std::stod(argv[9]) : 1.0;
					ovo_svm.setGamma(gamma);
				} else if (kernelType == "poly") {
					const double degree = (argc > 9) ? std::stod(argv[9]) : 1.0;
					const double coef0 = (argc > 10) ? std::stod(argv[10]) : 0.0;
					ovo_svm.setPolyParam(degree, coef0);
				}
				
				const auto start1 = std::chrono::steady_clock::now();
				
				if (parallel)
					ovo_svm.MPI_Fit(); // Parallel
				else
					ovo_svm.fit(); // Serial;
				
				const auto end1 = std::chrono::steady_clock::now();
				
				const auto duration1 = end1 - start1; // Время обучения модели
				
				const auto ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(duration1).count();
				const auto us1 = std::chrono::duration_cast<std::chrono::microseconds>(duration1).count();
				
				if (rank == 0) {
					if (parallel) {
						if (ms1 <= ACCURACY_MESURE)
							std::cout << "Execution time for training model (mpi): " << us1 / 1000.0 << " milliseconds\n";
						else
							std::cout << "Execution time for training model (mpi): " << ms1 << " milliseconds\n";
					} else {
						if (ms1 <= ACCURACY_MESURE)
							std::cout << "Execution time for training model (serial): " << us1 / 1000.0 << " milliseconds\n";
						else
							std::cout << "Execution time for training model (serial): " << ms1 << " milliseconds\n";
					}
					
				}
				
				if (flag2 || true) {
					
					autoscaling(x_test, n_test, m_test); // Шкалируем входные данные
					
					if (parallel) {
						const auto start2_1 = std::chrono::steady_clock::now();
						
						std::vector<std::string> yPred2_1 = ovo_svm.MPI_Predict1(x_test, n_test, m_test);
						
						const auto end2_1 = std::chrono::steady_clock::now();
						const auto duration2_1 = end2_1 - start2_1; // Время предсказания результата
						
						const auto ms2_1 = std::chrono::duration_cast<std::chrono::milliseconds>(duration2_1).count();
						const auto us2_1 = std::chrono::duration_cast<std::chrono::microseconds>(duration2_1).count();
						
						const double accuracy2_1 = getAccuracy(y_test, yPred2_1);
						const double balanced_accuracy2_1 = getBalancedAccuracy(y_test, yPred2_1);
						if (rank == 0) {	
							//std::cout << "Execution time for get predictions (mpi_first): " << ms2_1 << " milliseconds\n";
							
							if (ms2_1 <= ACCURACY_MESURE)
								std::cout << "Execution time for get predictions (mpi_first): " << us2_1 / 1000.0 << " milliseconds\n";
							else
								std::cout << "Execution time for get predictions (mpi_first): " << ms2_1 << " milliseconds\n";
							
							std::cout << "Accuracy of classification (mpi_first): " << accuracy2_1 << std::endl;
							std::cout << "Balanced accuracy of classification (mpi_first): " << balanced_accuracy2_1 << std::endl;
						}	
						
						const auto start2_2 = std::chrono::steady_clock::now();
						
						std::vector<std::string> yPred2_2 = ovo_svm.MPI_Predict2(x_test, n_test, m_test);
						
						const auto end2_2 = std::chrono::steady_clock::now();
						const auto duration2_2 = end2_2 - start2_2;
						
						const auto ms2_2 = std::chrono::duration_cast<std::chrono::milliseconds>(duration2_2).count();
						const auto us2_2 = std::chrono::duration_cast<std::chrono::microseconds>(duration2_2).count();
						
						const double accuracy2_2 = getAccuracy(y_test, yPred2_2);
						const double balanced_accuracy2_2 = getBalancedAccuracy(y_test, yPred2_2);
						if (rank == 0) {
							
							if (ms2_2 <= ACCURACY_MESURE)
								std::cout << "Execution time for get predictions (mpi_second): " << us2_2 / 1000.0 << " milliseconds\n";
							else
								std::cout << "Execution time for get predictions (mpi_second): " << ms2_2 << " milliseconds\n";
							
							std::cout << "Accuracy of classification (mpi_second): " << accuracy2_2 << std::endl;
							std::cout << "Balanced accuracy of classification (mpi_second): " << balanced_accuracy2_2 << std::endl;
						}
						
					} else {
						const auto start2_s = std::chrono::steady_clock::now();
						
						std::vector<std::string> yPred2_s = ovo_svm.predict(x_test, n_test, m_test);
						
						const auto end2_s = std::chrono::steady_clock::now();
						
						const auto duration2_s = end2_s - start2_s; // Время предсказания результата
						
						const auto ms2_s = std::chrono::duration_cast<std::chrono::milliseconds>(duration2_s).count();
						const auto us2_s = std::chrono::duration_cast<std::chrono::microseconds>(duration2_s).count();
						
						const double accuracy2_s = getAccuracy(y_test, yPred2_s);
						const double balanced_accuracy2_s = getBalancedAccuracy(y_test, yPred2_s);
						
						if (ms2_s <= ACCURACY_MESURE)
							std::cout << "Execution time for get predictions (serial): " << us2_s / 1000.0 << " milliseconds\n";
						else
							std::cout << "Execution time for get predictions (serial): " << ms2_s << " milliseconds\n";
						
						
						std::cout << "Accuracy of classification (serial): " << accuracy2_s << std::endl;
						std::cout << "Balanced accuracy of classification (serial): " << balanced_accuracy2_s << std::endl;
					}
					
				} else if (rank == 0) {
					std::cout << "Accuracy of classification: " << 1.0 << std::endl;
					std::cout << "Balanced accuracy of classification: " << 1.0 << std::endl;
				}
			} else {
				
				const double c = (argc > 4) ? std::stod(argv[4]) : 1.0;
				const double b = (argc > 5) ? std::stod(argv[5]) : 0.0;
				const double acc = (argc > 6) ? std::stod(argv[6]) : 0.0001;
				const int maxIt = (argc > 7) ? std::stoi(argv[7]) : 10000;
				std::string kernelType = (argc > 8) ? std::string(argv[8]) : "liner";
				
				ovoSVM ovo_svm(x_train, y_train, n_train, m_train);
				
				ovo_svm.setParameters(kernelType, c, b, acc, maxIt);
				if (kernelType == "rbf") {
					const double gamma = (argc > 9) ? std::stod(argv[9]) : 1.0;
					ovo_svm.setGamma(gamma);
				} else if (kernelType == "poly") {
					const double degree = (argc > 9) ? std::stod(argv[9]) : 1.0;
					const double coef0 = (argc > 10) ? std::stod(argv[10]) : 0.0;
					ovo_svm.setPolyParam(degree, coef0);
				}
				const auto start1 = std::chrono::steady_clock::now();
				
				if (parallel)
					ovo_svm.MPI_Fit(); // Parallel
				else
					ovo_svm.fit(); // Serial;
				
				const auto end1 = std::chrono::steady_clock::now();
				
				const auto duration1 = end1 - start1; // Время обучения модели
				
				const auto ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(duration1).count();
				const auto us1 = std::chrono::duration_cast<std::chrono::microseconds>(duration1).count();
				
				if (rank == 0) {
					if (parallel) {
						if (ms1 <= ACCURACY_MESURE)
							std::cout << "Execution time for training model (mpi): " << us1 / 1000.0 << " milliseconds\n";
						else
							std::cout << "Execution time for training model (mpi): " << ms1 << " milliseconds\n";
					} else {
						if (ms1 <= ACCURACY_MESURE)
							std::cout << "Execution time for training model (serial): " << us1 / 1000.0 << " milliseconds\n";
						else
							std::cout << "Execution time for training model (serial): " << ms1 << " milliseconds\n";
					}
					
				}
				
				autoscaling(x_test, n_test, m_test); // Шкалируем входные данные
				
				if (parallel) {
						const auto start2_1 = std::chrono::steady_clock::now();
						
						std::vector<std::string> yPred2_1 = ovo_svm.MPI_Predict1(x_test, n_test, m_test);
						
						const auto end2_1 = std::chrono::steady_clock::now();
						const auto duration2_1 = end2_1 - start2_1; // Время предсказания результата
						
						const auto ms2_1 = std::chrono::duration_cast<std::chrono::milliseconds>(duration2_1).count();
						const auto us2_1 = std::chrono::duration_cast<std::chrono::microseconds>(duration2_1).count();
						
						if (rank == 0) {
							if (ms2_1 <= ACCURACY_MESURE)
								std::cout << "Execution time for get predictions (mpi_first): " << us2_1 / 1000.0 << " milliseconds\n";
							else
								std::cout << "Execution time for get predictions (mpi_first): " << ms2_1 << " milliseconds\n";
						}
						
						const auto start2_2 = std::chrono::steady_clock::now();
						
						std::vector<std::string> yPred2_2 = ovo_svm.MPI_Predict2(x_test, n_test, m_test);
						
						const auto end2_2 = std::chrono::steady_clock::now();
						const auto duration2_2 = end2_2 - start2_2;
						
						const auto ms2_2 = std::chrono::duration_cast<std::chrono::milliseconds>(duration2_2).count();
						const auto us2_2 = std::chrono::duration_cast<std::chrono::microseconds>(duration2_2).count();
						
						if (rank == 0) {
							if (ms2_2 <= ACCURACY_MESURE)
								std::cout << "Execution time for get predictions (mpi_second): " << us2_2 / 1000.0 << " milliseconds\n";
							else
								std::cout << "Execution time for get predictions (mpi_second): " << ms2_2 << " milliseconds\n";
						}
					} else {
						const auto start2_s = std::chrono::steady_clock::now();
						
						std::vector<std::string> yPred2_s = ovo_svm.predict(x_test, n_test, m_test);
						
						const auto end2_s = std::chrono::steady_clock::now();
						
						const auto duration2_s = end2_s - start2_s; // Время предсказания результата
						
						const auto ms2_s = std::chrono::duration_cast<std::chrono::milliseconds>(duration2_s).count();
						const auto us2_s = std::chrono::duration_cast<std::chrono::microseconds>(duration2_s).count();
						
						if (ms2_s <= ACCURACY_MESURE)
							std::cout << "Execution time for get predictions (serial): " << us2_s / 1000.0 << " milliseconds\n";
						else
							std::cout << "Execution time for get predictions (serial): " << ms2_s << " milliseconds\n";
						
					}
			}
			
			deleting(x_train);
			deleting(x_test);
			
			deleting(y_train);
			deleting(y_test);
		
		}
	
	} catch (const std::exception& e) {
		std::cerr << "Error occurred: " << e.what() << std::endl;
	} catch (...) {
		std::cout << "Something went wrong\n"; 
	}
	
	MPI_Finalize();
	
	return 0;
}
