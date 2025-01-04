#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cxxplot/cxxplot>

// int main()
// {
//     double a = 0, b = 1000;
//     std::cout << a / b;
//     return 0;
// }

// // class DataLoader
// // {
// // public:
// //     DataLoader(std::string filename)
// //     {
// //         // dataset.open(filename.c_str());
// //         std::vector<std::vector<double>> data;
// //         std::fstream file(filename);
// //         std::string line;

// //         while (getline(file, line))
// //         {
// //             std::stringstream ss(line);
// //             std::string value;
// //             std::vector<double> row;

// //             while (getline(ss, value, ','))
// //             {
// //                 row.push_back(std::stod(value));
// //             }

// //             data.push_back(row);
// //         }

// //         for (int i{}; i < data[0].size(); ++i)
// //         {
// //             std::vector<double> col_vals;

// //             for (int j{}; j < data.size() - 1; ++j)
// //             {
// //                 col_vals.push_back(data[j][i]);
// //             }

// //             std::sort(col_vals.begin(), col_vals.end());
// //             int n = col_vals.size();
// //             double median = n % 2 != 0 ? col_vals[n / 2] : (col_vals[(n / 2) - 1] + col_vals[n / 2]) / 2;
// //             double iqr = col_vals[3 * n / 4] - col_vals[n / 4];

// //             for (int j{}; j < data.size(); ++j)
// //             {
// //                 data[j][i] = (data[j][i] - median) / iqr;
// //             }
// //         }

// //         std::ofstream out("./data/new_data.csv");

// //         for (int i{}; i < data.size(); ++i)
// //         {
// //             for (int j{}; j < data[i].size(); ++j)
// //             {
// //                 out << data[i][j];
// //                 if (j != data[i].size() - 1)
// //                 {
// //                     out << ',';
// //                 }
// //             }
// //             if (i != data.size() - 1)
// //             {
// //                 out << '\n';
// //             }
// //         }

// //         dataset.open("./data/new_data.csv");
// //     }

// // private:
// //     std::fstream dataset;
// // };

// // int main(int argc, char **argv)
// // {
// //     DataLoader dataset(argv[1]);
// //     return 0;
// // }

// // // #include <iostream>
// // // #include <fstream>
// // // #include <sstream>
// // // #include <vector>
// // // #include <string>

// // // int main() {
// // //     std::ifstream file("./data/pima-indians-diabetes.data.csv");
// // //     std::string line;
// // //     std::vector<std::vector<std::string>> data;
// // //     std::vector<std::string> target;

// // //     while (std::getline(file, line)) {
// // //         std::stringstream ss(line);
// // //         std::string value;
// // //         std::vector<std::string> row;

// // //         while (std::getline(ss, value, ',')) {
// // //             row.push_back(value);
// // //         }

// // //         if (!row.empty()) {
// // //             target.push_back(row.back());
// // //             row.pop_back();
// // //             data.push_back(row);
// // //         }
// // //     }

// // //     std::cout << "Input Data:" << std::endl;
// // //     for (const auto& row : data) {
// // //         for (const auto& val : row) {
// // //             std::cout << val << " ";
// // //         }
// // //         std::cout << std::endl;
// // //     }

// // //     std::cout << "Target Data:" << std::endl;
// // //     for (const auto& val : target) {
// // //         std::cout << val << std::endl;
// // //     }

// // //     return 0;
// // // }

// // //     // int get_input(std::vector<double> &input)
// // //     // {
// // //     //     input.clear();
// // //     //     std::string line;

// // //     //     std::vector<double> row;
// // //     //     // while (getline(dataset, line))
// // //     //     getline(dataset, line);
// // //     //     // {
// // //     //     // row.clear();
// // //     //     std::istringstream s(line);
// // //     //     std::string field;

// // //     //     while (std::getline(s, field, ','))
// // //     //     {
// // //     //         row.push_back(std::stod(field));
// // //     //     }
// // //     //     row.pop_back();
// // //     //     // for (int i{}; i < row.size(); ++i)
// // //     //     // {
// // //     //     //     std::cout << row[i];
// // //     //     //     i == row.size() - 1 ? std::cout << "\n" : std::cout << ", ";
// // //     //     // }
// // //     //     // }
// // //     //     return row.size();
// // //     // }

int main()
{
    std::vector<cxxplot::point2d> test_data = {
        {0.0, 0.1}, {1.0, 0.2}, {2.0, 0.3}, {3.0, 0.4}};
    cxxplot::plot(test_data);
    return 0;
}
