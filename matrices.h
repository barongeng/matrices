// (C) Craig Henderson
// For research use only
// hello@craighenderson.co.uk

#pragma once

#include "detail.h"

namespace matrices {

template<typename T, int Rows, int Cols, typename MatrixOrientation=RowOriented>
class matrix
{
  public:
    typedef T value_type;
    using determinant_value_type = typename detail::determinant_value_type<T>::type;
    using this_type = matrix<T, Rows, Cols, MatrixOrientation>;

    static int const cols = Cols;
    static int const rows = Rows;

  public:
    matrix()                                = default;
    matrix(this_type const &)               = default;
    this_type &operator=(this_type const &) = default;

    matrix(value_type const (&data)[Rows][Cols])
    {
        detail::assign<MatrixOrientation>(data_, data);
    }

    matrix(matrix<T, Rows, Cols, typename detail::inverse_orientation<MatrixOrientation>::type> const &other) : matrix()
    {
        *this = other;
    }
    
    matrix(this_type &&other) : data_(std::move(other.data_))
    {
    }

    matrix(T const &value) : matrix()
    {
        *this = value;
    }

    this_type &operator=(matrix<T, Rows, Cols, typename detail::inverse_orientation<MatrixOrientation>::type> const &other)
    {
        detail::assign<MatrixOrientation>(data_, array_type(other));
        return *this;
    }

    this_type &operator=(T const &operand)
    {
        // assignment is faster with vectorization that with threads
        for_each_noparallel([&operand](T &value) { value = operand; });
        return *this;
    }

    this_type &operator=(std::function<T (void)> fn)
    {
        // no threading; fn() may not be thread safe
        for_each_noparallel([&fn](T &value) { value = fn(); });
        return *this;
    }

    this_type &operator-=(T const &operand)
    {
        for_each_parallel([&operand](T &value) { value -= operand; });
        return *this;
    }

    this_type &operator+=(T const &operand)
    {
        for_each_parallel([&operand](T &value) { value += operand; });
        return *this;
    }

    this_type &operator*=(T const &operand)
    {
        for_each_parallel([&operand](T &value) { value *= operand; });
        return *this;
    }

    this_type &operator/=(T const &operand)
    {
        for_each_parallel([&operand](T &value) { value /= operand; });
        return *this;
    }

    this_type &operator-()
    {
        for_each_parallel([](T &value) { value = -value; });
        return *this;
    }

    template<typename Res=std::enable_if<std::is_same<MatrixOrientation,RowOriented>::value, T&>::type>
    Res at(int row, int col, RowOriented const& =MatrixOrientation())
    {
        return data_.get_at(row,col);
    }

    template<typename Res=std::enable_if<std::is_same<MatrixOrientation,ColumnOriented>::value, T&>::type>
    Res at(int row, int col, ColumnOriented const& =MatrixOrientation())
    {
        return data_.get_at(col,row);
    }

    template<typename Res=std::enable_if<std::is_same<MatrixOrientation,RowOriented>::value, T const &>::type>
    Res at(int row, int col, RowOriented const& =MatrixOrientation()) const
    {
        return data_.get_at(row,col);
    }

    template<typename Res=std::enable_if<std::is_same<MatrixOrientation,ColumnOriented>::value, T const &>::type>
    Res at(int row, int col, ColumnOriented const& =MatrixOrientation()) const
    {
        return data_.get_at(col,row);
    }

    template<typename Fn>
    void for_each_parallel(Fn fn)
    {
#pragma omp parallel for
        for (int j=0; j<Rows; ++j)
            for (int i=0; i<Cols; ++i)
                fn(at(j,i));
    }

    template<typename Fn>
    void for_each_noparallel(Fn fn)
    {
        for (int j=0; j<Rows; ++j)
            for (int i=0; i<Cols; ++i)
                fn(at(j,i));
    }

    static this_type identity()
    {
        static_assert(Rows == Cols, "Identity matrix is only valid for square matrices");

        this_type result = 0;
#pragma omp parallel for
        for (int j=0; j<Rows; ++j)
            result.at(j,j) = static_cast<T>(1);
        return result;
    }

    template<int R2, int C2, typename T, typename Fn>
    bool const is_equal(T const &other, Fn fn) const
    {
        static_assert(Rows == R2, "Matrices have different number of rows");
        static_assert(Cols == C2, "Matrices have different number of columns");

        for (int j=0; j<Rows; ++j)
        {
            for (int i=0; i<Cols; ++i)
            {
                if (at(j,i) != fn(other, j, i))
                    return false;
            }
        }
        return true;
    }

    template<typename T2, int R2, int C2, typename O2>
    bool const is_equal(matrix<T2,R2,C2,O2> const &other) const
    {
        static_assert(std::is_same<value_type, T2>::value, "Matrices have different types");

        return is_equal<R2, C2>(
            other,
            [](matrix<T2,R2,C2,O2> const &other, int j, int i) -> T2 {
                return other.at(j,i);
            });
    }

    template<typename T2, int R2, int C2>
    bool const is_equal(T2 const (&other)[R2][C2]) const
    {
        static_assert(std::is_same<value_type, T2>::value, "Matrices have different types");

        return is_equal<R2, C2>(
            other,
            [](T2 const (&other)[R2][C2], int j, int i) -> T2 {
                return other[j][i];
            });
    }

    determinant_value_type determinant()
    {
        static_assert(Rows == Cols, "Determinant is only valid for square matrices");

        // det(M) = det(L) * det(U), which for triangular matrices
        /// is just the product of the entries in their diagonal.
        auto lu = lu_decomposition();

        determinant_value_type product = static_cast<determinant_value_type>(1);
        for (int i=0; i<Rows; ++i)
        {
            product *= lu.first.at(i,i);
            product *= lu.second.at(i,i);
        }
        return product;
    }

    bool const is_singular() const
    {
        // using the rank, when I have is more efficient
        // http://trac.sagemath.org/attachment/ticket/12370/trac_12370_improve_is_singular.patch
        // return rank() != Rows;
        static_assert(Rows == Cols, "Singular is only valid for square matrices");
        return determinant() == static_cast<determinant_value_type>(0);
    }

    // from http://www.sanfoundry.com/cpp-program-perform-lu-decomposition-any-matrix/
    std::pair<
        matrix<determinant_value_type, Rows, Cols, MatrixOrientation>,
        matrix<determinant_value_type, Rows, Cols, MatrixOrientation>>
    lu_decomposition()
    {
        static_assert(Rows == Cols, "LU Decomposition is only valid for square matrices");
        int n = Rows;

        matrix<determinant_value_type, Rows, Cols, MatrixOrientation> l;
        matrix<determinant_value_type, Rows, Cols, MatrixOrientation> u;

        int i = 0, j = 0, k = 0;
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                if (j < i)
                    l.at(j,i) = 0;
                else
                {
                    l.at(j,i) = static_cast<determinant_value_type>(at(j,i));
                    for (k = 0; k < i; k++)
                    {
                        l.at(j,i) = l.at(j,i) - l.at(j,k) * u.at(k,i);
                    }
                }
            }

            for (j = 0; j < n; j++)
            {
                if (j < i)
                    u.at(i,j) = 0;
                else if (j == i)
                    u.at(i,j) = 1;
                else
                {
                    u.at(i,j) = at(i,j) / l.at(i,i);
                    for (k = 0; k < i; k++)
                    {
                        u.at(i,j) = u.at(i,j) - ((l.at(i,k) * u.at(k,j)) / l.at(i,i));
                    }
                }
            }
        }
        return { l, u };
    }

    template<typename Orientation=MatrixOrientation>
    void transpose_to(matrix<T, Cols, Rows, Orientation> &other) const;

    template<>
    void transpose_to(matrix<T, Cols, Rows, RowOriented> &other) const
    {
        other = matrix<T, Cols, Rows, MatrixOrientation>(data_.transpose());
    }

    template<typename Res=typename std::enable_if<Rows == Cols, matrix<T, Rows, Cols, MatrixOrientation>>::type>
    Res transpose()
    {
        using std::swap;
        for (int j=0; j<Rows; ++j)
        {
            for (int i=j+1; i<Cols; ++i)
                swap(at(i,j), at(j,i));
        }

        return *this;
    }

    template<>
    void transpose_to(matrix<T, Cols, Rows, ColumnOriented> &other) const
    {
        // initialising a ColumnOriented matrix from an array will always
        // change the orientation, because native arrays are RowOriented,
        // so we don't need to call transpose(), just let the init ctor
        // of matrix do the work
        other = matrix<T, Cols, Rows, MatrixOrientation>(data_);
    }

    friend std::ostream &operator<<(std::ostream &os, this_type const &matrix)
    {
        for (int j=0; j<Rows; ++j)
        {
            for (int i=0; i<Cols; ++i)
                os << matrix.at(j,i) << ' ';
            os << '\n';
        }
        return os;
    }

    friend class matrix<T, Rows, Cols, typename detail::inverse_orientation<MatrixOrientation>::type>;

  private:
    typedef T const (&array_type)[Rows][Cols];
    operator array_type() const
    {
        return data_;
    }

  private:
    using matrix_data = typename detail::vector_oriented_matrix_data<T, Rows, Cols, MatrixOrientation>::value_type;
    matrix_data data_;
};

template<typename T, int I, int J, int K, typename O1, typename O2>
matrix<T,I,K,RowOriented>
multiply(matrix<T,I,J,O1> const &a, matrix<T,J,K,O2> const &b)
{
    matrix<T,I,K,RowOriented> result;
#pragma omp parallel for
    for (int l=0; l<I; ++l)
    {
        for (int m=0; m<K; ++m)
        {
            T value = T();
            for (int c=0; c<J; ++c)
            {
                value += a.at(l,c) * b.at(c,m);
            }
            result.at(l,m) = value;
        }
    }
    return result;
}

template<typename T, int I, int J, int K, typename O1, typename O2>
matrix<T,I,K,RowOriented>
operator*(matrix<T,I,J,O1> const &a, matrix<T,J,K,O2> const &b)
{
    return multiply(a, b);
}

template<typename T1, int R1, int C1, typename O1, typename T2, int R2, int C2, typename O2>
bool const
operator==(matrix<T1,R1,C1,O1> const &a, matrix<T2,R2,C2,O2> const &b)
{
    return a.is_equal(b);
}

template<typename T1, int R1, int C1, typename O1, typename T2, int R2, int C2>
bool const
operator==(matrix<T1,R1,C1,O1> const &a, T2 const (&b)[R2][C2])
{
    return a.is_equal(b);
}

template<typename T1, int R1, int C1, typename T2, int R2, int C2, typename O2>
bool const
operator==(T1 const (&a)[R1][C1], matrix<T2,R2,C2,O2> const &b)
{
    return b.is_equal(a);
}

template<typename T1, int R1, int C1, typename O1, typename T2, int R2, int C2, typename O2>
bool const
operator!=(matrix<T1,R1,C1,O1> const &a, matrix<T2,R2,C2,O2> const &b)
{
    return !(a == b);
}

template<typename T, int R, int C, typename O>
matrix<T,R,C,O>
operator-(matrix<T,R,C,O> a, T const &b)
{
    return a -= b;
}

template<typename T, int R, int C, typename O>
matrix<T,R,C,O>
operator+(matrix<T,R,C,O> a, T const &b)
{
    return a += b;
}

template<typename T, int R, int C, typename O>
matrix<T,R,C,O>
operator*(matrix<T,R,C,O> a, T const &b)
{
    return a *= b;
}

template<typename T, int R, int C, typename O>
matrix<T,R,C,O>
operator/(matrix<T,R,C,O> a, T const &b)
{
    return a /= b;
}

template<typename T, int Rows, int Cols>
matrix<T, Rows, Cols,RowOriented>
make_matrix(T const (&data)[Rows][Cols])
{
    return matrix<T, Rows, Cols,RowOriented>(data);
}

template<typename T, int Rows, int Cols>
matrix<T, Rows, Cols,ColumnOriented>
make_column_oriented_matrix(T const (&data)[Rows][Cols])
{
    return matrix<T, Rows, Cols,ColumnOriented>(data);
}

template<typename T, int Rows, int Cols, typename MatrixOrientation>
matrix<T, Cols, Rows, MatrixOrientation>
transpose(matrix<T, Rows, Cols, MatrixOrientation> const &m)
{
    matrix<T, Cols, Rows, MatrixOrientation> result;
    m.transpose_to(result);
    return result;
}

}   // namespace matrices
