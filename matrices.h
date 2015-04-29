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
    typedef T type;
    using matrix_data = typename detail::vector_oriented_matrix_data<T, Rows, Cols, MatrixOrientation>::type;

    static int const cols = Cols;
    static int const rows = Rows;

  public:
    matrix() = default;

    matrix(type const (&data)[Rows][Cols])
    {
        detail::assign<T, Rows, Cols, MatrixOrientation>()(data_, data);
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

    template<typename T2, int R2, int C2, typename O2>
    bool const is_equal(matrix<T2,R2,C2,O2> const &other) const
    {
        static_assert(std::is_same<type, T2>::value, "Matrices have different types");
        static_assert(Rows == R2, "Matrices have different number of rows");
        static_assert(Cols == C2, "Matrices have different number of columns");

        for (int j=0; j<Rows; ++j)
        {
            for (int i=0; i<Cols; ++i)
            {
                if (at(j,i) != other.at(j,i))
                    return false;
            }
        }
        return true;
    }

    friend std::ostream &operator<<(std::ostream &os, matrix<T, Rows, Cols, MatrixOrientation> const &matrix)
    {
        for (int j=0; j<Rows; ++j)
        {
            for (int i=0; i<Cols; ++i)
                os << matrix.at(j,i) << ' ';
            os << '\n';
        }
        return os;
    }

  private:
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

template<typename T1, int R1, int C1, typename O1, typename T2, int R2, int C2, typename O2>
bool const
operator!=(matrix<T1,R1,C1,O1> const &a, matrix<T2,R2,C2,O2> const &b)
{
    return !(a == b);
}

template<typename T, int Rows, int Cols>
matrix<T,Rows,Cols,RowOriented>
make_matrix(T const (&data)[Rows][Cols])
{
    return matrix<T,Rows,Cols,RowOriented>(data);
}

template<typename T, int Rows, int Cols>
matrix<T,Rows,Cols,ColumnOriented>
make_column_oriented_matrix(T const (&data)[Rows][Cols])
{
    return matrix<T,Rows,Cols,ColumnOriented>(data);
}

}   // namespace matrices
