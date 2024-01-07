#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

/* Sequential */
std::vector<std::vector<int>>
sequential_matrix_multiply(const std::vector<std::vector<int>> &a,
                           const std::vector<std::vector<int>> &b) {
  size_t rows_a = a.size();
  size_t cols_a = a[0].size();
  size_t cols_b = b[0].size();

  std::vector<std::vector<int>> result(rows_a, std::vector<int>(cols_b, 0));

  for (size_t i = 0; i < rows_a; ++i) {
    for (size_t j = 0; j < cols_b; ++j) {
      for (size_t k = 0; k < cols_a; ++k) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

void multipy_block(const std::vector<std::vector<int>> &a,
                   const std::vector<std::vector<int>> &b,
                   std::vector<std::vector<int>> &result, int start_row,
                   int end_row) {

  size_t cols_a = a[0].size();
  size_t cols_b = b[0].size();

  for (size_t i = static_cast<size_t>(start_row);
       i < static_cast<size_t>(end_row); ++i) {
    for (size_t j = 0; j < cols_b; ++j) {
      int sum = 0;
      for (size_t k = 0; k < cols_a; ++k) {
        sum += a[i][k] * b[k][j];
      }

      // Use mutex to safely update the result matrix
      std::lock_guard<std::mutex> lock(std::mutex result_mutex);
      result[i][j] += sum;
    }
  }
}

/* multiply matrix in bloc parallel  */
void block_matrix_multiplication(const std::vector<std::vector<int>> &a,
                                 const std::vector<std::vector<int>> &b,
                                 std::vector<std::vector<int>> &result,
                                 int num_threads) {
  size_t rows_a = a.size();
  int rows_per_thread = static_cast<int>(rows_a) / num_threads;
  int extra_rows = static_cast<int>(rows_a) % num_threads;

  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    int start_row = i * rows_per_thread + std::min(i, extra_rows);
    int end_row = (i + 1) * rows_per_thread + std::min(i + 1, extra_rows);

    threads.emplace_back(multipy_block, std::ref(a), std::ref(b),
                         std::ref(result), start_row, end_row);
  }
  for (std::thread &t : threads) {
    t.join();
  }
}

/* multiply matrix in bloc parallel  */
void modulo_matrix_multiplication(const std::vector<std::vector<int>> &a,
                                  const std::vector<std::vector<int>> &b,
                                  std::vector<std::vector<int>> &result,
                                  int num_threads) {
  size_t rows_a = a.size();
  int rows_per_thread = static_cast<int>(rows_a) / num_threads;

  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    int start_row = i * rows_per_thread +
                    std::min(i, static_cast<int>(rows_a) % num_threads);
    int end_row = (i + 1) * rows_per_thread +
                  std::min(i + 1, static_cast<int>(rows_a) % num_threads);

    threads.emplace_back(multipy_block, std::ref(a), std::ref(b),
                         std::ref(result), start_row, end_row);
  }
  for (std::thread &t : threads) {
    t.join();
  }
}

void display_matrix(const std::vector<std::vector<int>> &matrix) {
  for (const auto &row : matrix) {
    for (int value : row) {
      std::cout << value << "\t";
    }
    std::cout << "\n";
  }
}

/* void modulo_matrix_parallel_multiplication() */

int main() {

  /* cannot use this. the data is so small that the cost of parallelism is
   * obvious */
  /* std::vector<std::vector<int>> matrix_a = {{1, 2, 3}, {4, 5, 6}}; */
  /* std::vector<std::vector<int>> matrix_b = {{7, 8}, {9, 10}, {11, 12}}; */

  /* Creating two large empty matrices  */
  size_t rows_a = 5001;
  size_t cols_a = 500;
  size_t rows_b = 500;
  size_t cols_b = 1000;

  std::vector<std::vector<int>> matrix_a(rows_a, std::vector<int>(cols_a, 0));
  std::vector<std::vector<int>> matrix_b(rows_b, std::vector<int>(cols_b, 0));

  auto sequential_start_time = std::chrono::high_resolution_clock::now();

  std::vector<std::vector<int>> sequential_matrix =
      sequential_matrix_multiply(matrix_a, matrix_b);

  auto sequential_end_time = std::chrono::high_resolution_clock::now();

  auto sequential_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          sequential_end_time - sequential_start_time);

  std::cout << "Sequential Result Time: \n";
  std::cout << "Done! Time Taken : " << sequential_duration.count()
            << " miliseconds"
            << "\n";
  /* display_matrix(sequential_matrix); */

  int num_threads = static_cast<int>(std::thread::hardware_concurrency());
  std::vector<std::vector<int>> result_matrix(
      matrix_a.size(), std::vector<int>(matrix_b[0].size(), 0));

  auto block_start_time = std::chrono::high_resolution_clock::now();

  /* multiply matrix in bloc parallel  */

  block_matrix_multiplication(matrix_a, matrix_b, result_matrix, num_threads);

  /* std::cout << "Block Resultant Matrix: \n"; */
  auto block_end_time = std::chrono::high_resolution_clock::now();

  auto block_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      block_end_time - block_start_time);

  std::cout << "Block Result Time: \n";
  std::cout << "Done! Time Taken : " << block_duration.count() << " miliseconds"
            << "\n";
  /* display_matrix(result_matrix); */

  auto modulo_start_time = std::chrono::high_resolution_clock::now();

  /* multiply matrix in modulo parallel  */

  block_matrix_multiplication(matrix_a, matrix_b, result_matrix, num_threads);

  /* std::cout << "Block Resultant Matrix: \n"; */
  auto modulo_end_time = std::chrono::high_resolution_clock::now();

  auto modulo_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      modulo_end_time - modulo_start_time);

  std::cout << "Modulo Results Time: \n";
  std::cout << "Done! Time Taken : " << modulo_duration.count()
            << " miliseconds"
            << "\n";

  return 0;
}
