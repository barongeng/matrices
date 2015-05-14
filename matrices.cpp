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

void big_matrix_test()
{
    matrix<int, 13000, 4000, RowOriented>    m20r = 0;
    matrix<int, 13000, 4000, ColumnOriented> m20c = 0;
    test_matrix_operators("RowOriented   ", m20r);
    test_matrix_operators("ColumnOriented", m20c);
    check_and_throw(m20r == m20c);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 6);

#ifdef NDEBUG
    unsigned const DIM1 = 10000;
    unsigned const DIM2 = 4000;
    unsigned const DIM3 = 20000;
    typedef float element_t;
#else
    unsigned const DIM1 = 100;
    unsigned const DIM2 = 40;
    unsigned const DIM3 = 200;
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
}

void test_linear_systems()
{
    // Jeffrey, A. Essential of Engineering Mathematics
    // Example 56.1 - non-homgeneous
    double A1[3][3] = { { 1,2,3 }, { 1,3,5 }, { 1,5,12 } };
    double b1[3][1] = { 4,1,2 };
    assert(!matrices::make_matrix(A1).is_singular());
    std::cout << matrices::make_matrix(A1) << '\n' << matrices::make_matrix(b1) << '\n';
    std::cout << solve(matrices::make_matrix(A1), matrices::make_matrix(b1)) << '\n';

    // Jeffrey, A. Essential of Engineering Mathematics
    // Example 56.2 - non-homgeneous
    double A2[3][3] = { { 2,3,4 }, { 5,6,7 }, { 8,9,10 } };
    double b2[3][1] = { 1,2,4 };
    assert(matrices::make_matrix(A2).is_singular());
    std::cout << matrices::make_matrix(A2) << '\n' << matrices::make_matrix(b2) << '\n';
    std::cout << solve(matrices::make_matrix(A2), matrices::make_matrix(b2)) << '\n';

    // Jeffrey, A. Essential of Engineering Mathematics
    // Example 56.3 - non-homgeneous
    double A3[3][3] = { { 2,4,1 }, { 3,5,0 }, { 5,13,7 } };
    double b3[3][1] = { 1,1,4 };
    assert(matrices::make_matrix(A3).is_singular());
    std::cout << matrices::make_matrix(A3) << '\n' << matrices::make_matrix(b3) << '\n';
    std::cout << solve(matrices::make_matrix(A3), matrices::make_matrix(b3)) << '\n';

    // Jeffrey, A. Essential of Engineering Mathematics
    // Example 56.5 - homgeneous
    double A4[3][3] = { { 1,-2,3 }, { 2,4,5 }, { 1,2,6 } };
    double b4[3][1] = { 0,0,0 };
    assert(!matrices::make_matrix(A4).is_singular());
    std::cout << matrices::make_matrix(A4) << '\n' << matrices::make_matrix(b4) << '\n';
    std::cout << solve(matrices::make_matrix(A4), matrices::make_matrix(b4)) << '\n';

    // Jeffrey, A. Essential of Engineering Mathematics
    // Example 56.6 - homgeneous
    double A5[3][3] = { { 1,5,3 }, { 5,1,-1 }, { 1,2,1} };
    double b5[3][1] = { 0,0,0 };
    assert(matrices::make_matrix(A5).is_singular());
    std::cout << matrices::make_matrix(A5) << '\n' << matrices::make_matrix(b5) << '\n';
    std::cout << solve(matrices::make_matrix(A5), matrices::make_matrix(b5)) << '\n';

    // example from http://college.cengage.com/mathematics/larson/elementary_linear/4e/shared/downloads/c10s1.pdf
    // demonstrating rounding errors
    double M1[3][3] = { { 0.143, 0.357, 2.01 }, { -1.31, 0.911, 1.99 }, { 11.2, -4.3, -0.605 } };
    double r1[3][1] = { -5.17, -5.46, 4.42 };
    std::cout << matrices::make_matrix(M1) << '\n' << matrices::make_matrix(r1) << '\n';
    std::cout << solve(matrices::make_matrix(M1), matrices::make_matrix(r1)) << '\n';

    double M2[2][2] = { { 1, 1 }, { 1., 401./400. } };
    double r2[2][1] = { 0, 20 };
    std::cout << matrices::make_matrix(M2) << '\n' << matrices::make_matrix(r2) << '\n';
    std::cout << solve(matrices::make_matrix(M2), matrices::make_matrix(r2)) << '\n';
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

    std::cout << m10 << '\n' << transpose(m10) << '\n';
    std::cout << m18 << '\n' << transpose(m18) << '\n';

    long double lu_test_data[3][3] = { {6, 1, 1}, { 4, -2, 5 }, { 2, 8, 7 } };
    //int lu_test_data[3][3] = { {1, 1, -1}, { 2, -1, 3 }, { 3, 1, -1 } };
    auto lu_test = make_matrix(lu_test_data);
    std::cout << lu_test << '\n';
    auto lu = lu_test.lu_decomposition();
    std::cout << lu.first << '\n';
    std::cout << lu.second << '\n';
    assert(lu_test.determinant() == -306);

    std::cout << lu_test << '\n';
    std::cout << lu_test.transpose() << '\n';
    auto lu_test2 = make_column_oriented_matrix(lu_test_data);
    std::cout << lu_test2 << '\n';
    std::cout << lu_test2.transpose() << '\n';

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

    // test matrix from https://people.richland.edu/james/lecture/m116/matrices/pivot.html
    double linear_system_lhs[3][3] = { {3, 2, -4}, { 2, 3, 3 }, { 5, -3, 1 } };
    double linear_system_rhs[3][1] = { 3, 15, 14};
    auto solution = solve(make_matrix(linear_system_lhs), make_matrix(linear_system_rhs));
    std::cout << solution << '\n';
}

void test_transpose()
{
    int a[][3] = { { 1, 2 }, { 3, 4 } };
    auto m1 = make_matrix(a);
    auto at1 = transpose(m1);

    int b[][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
    auto b1 = make_matrix(b);
    auto bt1 = transpose(b1);

    matrix<int, 3, 2, RowOriented> bt2;
    b1.transpose_to(bt2);
    std::cout << b1 << '\n' << bt2 << '\n';

    matrix<int, 3, 2, ColumnOriented> bt3;
    b1.transpose_to(bt3);
    std::cout << b1 << '\n' << bt3 << '\n';


    int c[][2] = { { 7, 8 }, { 9, 10 }, { 11, 12} };
    auto c1 = make_column_oriented_matrix(c);
    auto ct1 = transpose(c1);

    matrix<int, 2, 3, RowOriented> ct2;
    c1.transpose_to(ct2);
    std::cout << c1 << '\n' << ct2 << '\n';

    matrix<int, 2, 3, ColumnOriented> ct3;
    c1.transpose_to(ct3);
    std::cout << c1 << '\n' << ct3 << '\n';


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 6);

    unsigned const DIM1 = 10000;
    unsigned const DIM2 = 4000;
    unsigned const DIM3 = 20000;
    float (&d)[DIM1][DIM2] = (float (&)[DIM1][DIM2])*(new float[DIM1][DIM2]);
    for (int j=0; j<DIM1; ++j)
        for (int i=0; i<DIM2; ++i)
            d[j][i] = float(dis(gen));
    auto d1  = make_matrix(d);
    auto dt1 = transpose(d1);
    assert(transpose(dt1) == d1);

    auto d2 = make_column_oriented_matrix(d);
    auto dt2 = transpose(d2);
    assert(transpose(dt2) == d2);

    assert(dt1 == dt2);
    assert(transpose(dt1) == transpose(dt2));
}

}   // namespace matrices



int main(int argc, char *argv[])
{
    matrices::test();
    matrices::test_transpose();
    matrices::test_linear_systems();
    matrices::big_matrix_test();
	return 0;
}

