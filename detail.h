// (C) Craig Henderson
// For research use only
// hello@craighenderson.co.uk

#pragma once

namespace matrices {

struct RowOriented    {};
struct ColumnOriented {};

namespace detail {

template<typename T, int Rows, int Cols>
class matrix_data
{
  public:
    typedef T value_type;
    typedef T const (&array_type)[Rows][Cols];

    matrix_data() : data_(new value_type[Rows*Cols])
    {
    }

    matrix_data(matrix_data const &other) : matrix_data()
    {
        *this = other;
    }

    matrix_data(matrix_data &&other)
    {
        swap(data_, other.data_);
    }

    matrix_data &operator=(matrix_data const &other)
    {
        using std::copy;
        copy(
            other.data_.get(),
            other.data_.get()+(Rows*Cols)+1,
            data_.get());
        return *this;
    }

    operator array_type() const
    {
        return reinterpret_cast<array_type>(*data_.get());
    }

    void copy(value_type const (&data)[Rows][Cols])
    {
        value_type *ptr = data_.get();
        for (auto const &d1 : data)
            for (auto const &d2 : d1)
                *ptr++ = d2;
    }

    void change_orientation(value_type const (&data)[Cols][Rows])
    {
        value_type *ptr = data_.get();
        for (int j=0; j<Rows; j++)
            for (int i=0; i<Cols; i++)
                *ptr++ = data[i][j];
    }

    friend class matrix_data<T, Cols, Rows>;
    matrix_data<T, Cols, Rows>
    transpose() const
    {
        matrix_data<T, Cols, Rows> result;

        value_type      *dst = result.data_.get();
        array_type const src = *this;
        for (int i=0; i<Cols; i++)
            for (int j=0; j<Rows; j++)
                *dst++ = src[j][i];

        return result;
    }

    value_type &get_at(int j, int i)
    {
        assert(j >=0  &&  j<Rows);
        assert(i >=0  &&  i<Cols);
        return data_.get()[j*Cols+i];
    }

    value_type const &get_at(int j, int i) const
    {
        assert(j >=0  &&  j<Rows);
        assert(i >=0  &&  i<Cols);
        return data_.get()[j*Cols+i];
    }

  private:
    std::unique_ptr<value_type[]> data_;
};



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



template<typename T>
struct determinant_value_type
{
    typedef float type;
};

template<>
struct determinant_value_type<double>
{
    typedef double type;
};

template<>
struct determinant_value_type<long double>
{
    typedef long double type;
};



template<typename T, int Rows, int Cols, typename MatrixOrientation>
struct vector_oriented_matrix_data;

template<typename T, int Rows, int Cols>
struct vector_oriented_matrix_data<T, Rows, Cols, RowOriented>
{
    typedef matrix_data<T,Rows,Cols> value_type;
};

template<typename T, int Rows, int Cols>
struct vector_oriented_matrix_data<T, Rows, Cols, ColumnOriented>
{
    typedef matrix_data<T,Cols,Rows> value_type;
};


template<typename T1, typename T2, bool ChangeOrientation>
struct assigner;

template<typename T1, typename T2>
struct assigner<T1, T2, false>
{
    void operator()(T1 &dest, T2 const &data)
    {
        dest.copy(data);
    }
};

template<typename T1, typename T2>
struct assigner<T1, T2, true>
{
    void operator()(T1 &dest, T2 const &data)
    {
        dest.change_orientation(data);
    }
};

template<typename Orientation, typename T1, typename T2, int Rows, int Cols>
void assign(
    T1 &dest,
    T2 const (&data)[Rows][Cols])
{
    assigner<T1, T2 const (&)[Rows][Cols], !std::is_same<Orientation, RowOriented>::value>()(dest, data);
}

}   // namespace detail

}   // namespace matrices
