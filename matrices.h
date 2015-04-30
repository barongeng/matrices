// (C) Craig Henderson
// For research use only
// hello@craighenderson.co.uk

#pragma once

#include "detail.h"

namespace matrices {

template<typename MatrixOrientation>
struct inverse_orientation;

template<>
struct inverse_orientation<ColumnOriented>
{
    typedef RowOriented type;
};

template<>
struct inverse_orientation<RowOriented>
{
    typedef ColumnOriented type;
};

template<typename T, int Rows, int Cols, typename MatrixOrientation=RowOriented>
class matrix
{
  public:
    typedef T value_type;
    using this_type   = matrix<T, Rows, Cols, MatrixOrientation>;

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

    matrix(matrix<T, Rows, Cols, typename inverse_orientation<MatrixOrientation>::type> const &other) : matrix()
    {
        *this = other;
    }
    
    matrix(this_type &&other) : data_(std::move(other.data_))
    {
    }

    this_type &operator=(matrix<T, Rows, Cols, typename inverse_orientation<MatrixOrientation>::type> const &other)
    {
        detail::assign<MatrixOrientation>(data_, array_type(other));
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

    template<typename Orientation=MatrixOrientation>
    matrix<T, Cols, Rows, Orientation>
    transpose() const;

    template<>
    matrix<T, Cols, Rows, RowOriented>
    transpose() const
    {
        return matrix<T, Cols, Rows, MatrixOrientation>(data_.transpose());
    }

    template<>
    matrix<T, Cols, Rows, ColumnOriented>
    transpose() const
    {
        // initialising a ColumnOriented matrix from an array will always
        // change the orientation, because native arrays are RowOriented,
        // so we don't need to call transpose(), just let the init ctor
        // of matrix do the work
        return matrix<T, Cols, Rows, MatrixOrientation>(data_);
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

    friend class matrix<T, Rows, Cols, typename inverse_orientation<MatrixOrientation>::type>;

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

}   // namespace matrices
