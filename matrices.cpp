// (C) Craig Henderson
// For research use only
// hello@craighenderson.co.uk

#include "stdafx.h"
#include <cassert>
#include <functional>
#include <memory>
#include <iostream>
#include <iomanip>
#include <exception>

// for testing
#include <random>
#include <chrono>

#include "matrices.h"

namespace matrices {

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
    assert(m10 == d);
    assert(d == m11);
    assert(m10 == m12);
    assert(m10 == m13);
    assert(m10 == m14);

    assert(m15 == d);
    assert(d == m16);
    assert(m15 == m17);
    assert(m15 == m18);

    assert(m15 == m10);
    assert(m10 == m15);
    std::cout << m18 << '\n';

    std::cout << m10 << '\n' << m10.transpose() << '\n';
    std::cout << m18 << '\n' << m18.transpose() << '\n';


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

