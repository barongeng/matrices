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

    matrix_data &operator=(matrix_data const &) = delete;

  public:
    matrix_data() : data_(new value_type[Rows*Cols])
    {
    }

    matrix_data(value_type const (&data)[Rows][Cols]) : matrix_data()
    {
        copy(data);
    }

    matrix_data(matrix_data &&other) : data_(std::move(other.data_))
    {
    }

    matrix_data(matrix_data const &other) : matrix_data()
    {
        using std::copy;
        copy(
            other.data_.get(),
            other.data_.get()+(Rows*Cols)+1,
            data_.get());
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


template<typename T, int Rows, int Cols, typename Orientation>
struct assign;

template<typename T, int Rows, int Cols>
struct assign<T,Rows,Cols,RowOriented>
{
    void operator()(typename detail::vector_oriented_matrix_data<T, Rows, Cols, RowOriented>::value_type &dest, T const (&data)[Rows][Cols])
    {
        dest.copy(data);
    }
};

template<typename T, int Rows, int Cols>
struct assign<T,Rows,Cols,ColumnOriented>
{
    void operator()(typename detail::vector_oriented_matrix_data<T, Rows, Cols, ColumnOriented>::value_type &dest, T const (&data)[Rows][Cols])
    {
        dest.change_orientation(data);
    }
};

}   // namespace detail

}   // namespace matrices
