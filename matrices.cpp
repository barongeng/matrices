// (C) Craig Henderson
// For research use only
// hello@craighenderson.co.uk

#include "stdafx.h"
#include <cassert>
#include <functional>
#include <string>
#include <memory>
#include <iostream>
#include <iomanip>
#include <exception>

// for testing
#include <random>
#include <chrono>

#include "matrices.h"

namespace matrices {

template<typename Fn>
void timer(std::string const &msg, Fn fn)
{
    std::cout << msg << ": ";
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
}

template<typename Matrix>
void test_matrix_operators(std::string const &msg, Matrix &m)
{
    timer(msg + " matrix Assignment         ", [&]{ m = 189; });
    timer(msg + " matrix Addition           ", [&]{ m + 11; });
    timer(msg + " matrix Self Addition      ", [&]{ m += 11; });
    timer(msg + " matrix Subtraction        ", [&]{ m - 11; });
    timer(msg + " matrix Self Subtraction   ", [&]{ m -= 11; });
    timer(msg + " matrix Division           ", [&]{ m / 4; });
    timer(msg + " matrix Self Division      ", [&]{ m /= 2; });
    timer(msg + " matrix Multiplication     ", [&]{ m * 2; });
    timer(msg + " matrix Self Multiplication", [&]{ m *= 2; });
    //timer(msg + " matrix muliply by itself, w/ threads", [&m]{ multiply(m, m.transpose());   });
}

void check_and_throw(bool const value)
{
    assert(value);
    if (!value)
        throw std::runtime_error("ASSERTion failed.");
}

void test()
{
    int a[][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
    int b[][2] = { { 7, 8 }, { 9, 10 }, { 11, 12} };
    auto m1 = make_matrix(a);
    auto m2 = make_matrix(b);
    auto m3 = multiply(m1, m2);
    std::cout << m1 << '\n';
    std::cout << m2 << '\n';
    std::cout << m3 << '\n';

    int c[][3] = { { 3, 4, 2 } };
    int d[][4] = { { 13, 9, 7, 15}, { 8, 7, 4, 6}, { 6, 4, 0, 3} };
    auto m4 = make_matrix(c);
    auto m5 = make_matrix(d);
    auto m6 = m4 * m5;
    std::cout << m6 << '\n';



    auto m7 = make_column_oriented_matrix(a);
    auto m8 = make_column_oriented_matrix(b);
    auto m9 = m7 * m8;
    std::cout << m7 << '\n';
    std::cout << m8 << '\n';
    std::cout << m9 << '\n';


    matrix<int, 3, 4, RowOriented> m10(d);              // data init ctor
    matrix<int, 3, 4, RowOriented> m11(m10);            // copy ctor
    matrix<int, 3, 4, RowOriented> m12 = m10;
    matrix<int, 3, 4, RowOriented> m13 { m10 };
    matrix<int, 3, 4, RowOriented> m14 = { m10 };
    std::cout << m10 << '\n';

    matrix<int, 3, 4, ColumnOriented> m15(m10);         // template copy ctor
    matrix<int, 3, 4, ColumnOriented> m16 = m10;
    matrix<int, 3, 4, ColumnOriented> m17 { m10 };
    matrix<int, 3, 4, ColumnOriented> m18 = { m10 };
    m18 = m10;
    check_and_throw(m10 == d);
    check_and_throw(d == m11);
    check_and_throw(m10 == m12);
    check_and_throw(m10 == m13);
    check_and_throw(m10 == m14);

    check_and_throw(m15 == d);
    check_and_throw(d == m16);
    check_and_throw(m15 == m17);
    check_and_throw(m15 == m18);

    check_and_throw(m15 == m10);
    check_and_throw(m10 == m15);
    std::cout << m18 << '\n';

    std::cout << m10 << '\n' << m10.transpose() << '\n';
    std::cout << m18 << '\n' << m18.transpose() << '\n';

    long double lu_test_data[3][3] = { {6, 1, 1}, { 4, -2, 5 }, { 2, 8, 7 } };
    //int lu_test_data[3][3] = { {1, 1, -1}, { 2, -1, 3 }, { 3, 1, -1 } };
    auto lu_test = make_matrix(lu_test_data);
    std::cout << lu_test << '\n';
    auto lu = lu_test.lu_decomposition();
    std::cout << lu.first << '\n';
    std::cout << lu.second << '\n';
    std::cout << lu_test.determinant() << '\n';

    std::cout << matrix<double, 4, 4, ColumnOriented>::identity() << '\n';
    std::cout << matrix<double, 4, 4, RowOriented>::identity().transpose() << '\n';

    matrix<int, 8, 8, RowOriented> m19 = 77;
    std::cout << m19 << '\n';
    m19 = []() -> int {
        static int i=0;
        return ++i;
    };
    std::cout << m19 << '\n';
    check_and_throw((m19 * matrix<int, 8, 8, RowOriented>::identity()) == m19);
    check_and_throw((m19 * matrix<int, 8, 8, ColumnOriented>::identity()) == m19);

    m19 -= 9;
    std::cout << m19 << '\n';
    std::cout << -m19 << '\n';
    m19 = m19 - 17;
    std::cout << m19 << '\n';

    m19 += 7;
    std::cout << m19 << '\n';
    m19 = m19 + 29;
    std::cout << m19 << '\n';

    m19 *= 2;
    std::cout << m19 << '\n';
    m19 = m19 * 29;
    std::cout << m19 << '\n';

    m19 /= 2;
    std::cout << m19 << '\n';
    m19 = m19 / 29;
    std::cout << m19 << '\n';

    matrix<int, 13000, 4000, RowOriented>    m20r = 0;
    matrix<int, 13000, 4000, ColumnOriented> m20c = 0;
    test_matrix_operators("RowOriented   ", m20r);
    test_matrix_operators("ColumnOriented", m20c);
    check_and_throw(m20r == m20c);

    auto big_matrix_test = []() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 6);

#ifdef NDEBUG
        unsigned const DIM1 = 10000;
        unsigned const DIM2 = 4000;
        unsigned const DIM3 = 20000;
        typedef float element_t;
#else
        unsigned const DIM1 = 10000;
        unsigned const DIM2 = 4000;
        unsigned const DIM3 = 20000;
        typedef int element_t;
#endif

        element_t (&a)[DIM1][DIM2] = (element_t (&)[DIM1][DIM2])*(new element_t[DIM1][DIM2]);
        for (int j=0; j<DIM1; ++j)
            for (int i=0; i<DIM2; ++i)
                a[j][i] = element_t(dis(gen));

        
        element_t (&b)[DIM2][DIM3] = (element_t (&)[DIM2][DIM3])*(new element_t[DIM2][DIM3]);
        for (int j=0; j<DIM2; ++j)
            for (int i=0; i<DIM3; ++i)
                b[j][i] = element_t(dis(gen));

        auto m1 = make_matrix(a);
        auto m2 = make_matrix(b);
        auto multiply_row_oriented = [&m1,&m2]() -> matrix<element_t,DIM1,DIM3,RowOriented> {
            auto m = m1 * m2;
            return m;
        };

        auto m3 = make_column_oriented_matrix(a);
        auto m4 = make_column_oriented_matrix(b);
        auto multiply_column_oriented = [&m3,&m4]() -> matrix<element_t,DIM1,DIM3,RowOriented> {
            return m3 * m4;
        };

        auto multiply_mixed_orientation = [&m1,&m4]() -> matrix<element_t,DIM1,DIM3,RowOriented> {
            return m1 * m4;
        };

        auto time_it = [](std::function<matrix<element_t,DIM1,DIM3,RowOriented> (void)> fn) -> matrix<element_t,DIM1,DIM3,RowOriented> {
            auto start = std::chrono::high_resolution_clock::now();
            auto res = fn();
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
            return res;
        };

        std::cout << "Running mixed-orientation " << m1.cols << 'x' << m1.rows << " by " << m4.cols << 'x' << m4.rows << " multiplication: ";
        auto r3 = time_it(multiply_mixed_orientation);
        std::cout << "Running row-oriented " << m1.cols << 'x' << m1.rows << " by " << m2.cols << 'x' << m2.rows << " multiplication: ";
        auto r1 = time_it(multiply_row_oriented);
        std::cout << "Running column-oriented " << m3.cols << 'x' << m3.rows << " by " << m4.cols << 'x' << m4.rows << " multiplication: ";
        auto r2 = time_it(multiply_column_oriented);

        if (r1 != r2  &&  r2 != r3)
            throw std::runtime_error("all results are unequal");
        if (r1 != r2)
            throw std::runtime_error("r1 and r2 are not equal");
        if (r2 != r3)
            throw std::runtime_error("r2 and r3 are not equal");
     };
    big_matrix_test();
}

}   // namespace matrices

int main(int argc, char *argv[])
{
    matrices::test();
	return 0;
}

