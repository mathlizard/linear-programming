#include <iostream>
#include <fstream>
#include <iterator>
#include <memory> 
#include <assert.h>

#define ID(i,j,k) (( i * k ) + j)

// generic Matrix datatype, will be good later too
template <class T>
class Matrix {
private:
	// number of rows
	size_t dim_rows_;

	//number of columns
	size_t dim_columns_;

	// the data
	std::unique_ptr<T[]> matrix_;

public:
	// rw access to matrix elements
	T& operator()(size_t const &row, size_t const &col) {
		return matrix_[ID(row, col, dim_columns_)];
	}

	// copy assignment operator
	Matrix<T>& operator=(Matrix<T> const &m) {
		size_t j;
		matrix_.reset();
		dim_rows_ = m.dim_rows_;
		dim_columns_ = m.dim_columns_;
		matrix_ = std::make_unique<T[]>( dim_rows_ * dim_columns_ );
		assert(matrix_ != nullptr);
		for (size_t i = 0; i < dim_rows_; ++i)
			for (j = 0; j < dim_columns_; ++j)
				matrix_[ID(i,j,dim_columns_)] = m.matrix_[ID(i,j,dim_columns_)];
		return *this;
	}

	// move assigment operator
	Matrix<T>& operator=(Matrix<T> &&m) {
		std::swap(this->matrix_, m.matrix_);
		std::swap(this->dim_columns_, m.dim_columns_);
		std::swap(this->dim_rows_, m.dim_rows_);
		return *this;
	}

	// constructor without fill value
	explicit Matrix(size_t const &a, size_t const &b) : dim_rows_(a), dim_columns_(b) {
		assert(a > 0 && b > 0);
		matrix_ = std::make_unique< T[] >( dim_rows_ * dim_columns_ );
		assert(matrix_ != nullptr);
	}

	// constructor fills the matrix with fill_value
	explicit Matrix(size_t const &a, size_t const &b, T const &fill_value) : Matrix(a, b) {
		size_t j;
		for (size_t i = 0; i < dim_rows_; ++i)
			for (j = 0; j < dim_columns_; ++j)
				matrix_[ID(i, j, dim_columns_)] = fill_value;
	}

	// default constructor
	Matrix() : dim_rows_(0), dim_columns_(0), matrix_(nullptr) { }

	// move constructor
	Matrix(Matrix<T> &&m) {
		std::swap(this->matrix_, m.matrix_);
		std::swap(this->dim_columns_, m.dim_columns_);
		std::swap(this->dim_rows_, m.dim_rows_);
	}

	bool empty() const {
		return (matrix_ == nullptr);
	}

	// destructor
	~Matrix() = default;

	// iterator: nested class, the necessary typedefs are being inherited
	class iterator: public std::iterator< std::forward_iterator_tag, T, T, T const*, T > {
	private:
		size_t dim_columns_;
		size_t row_index_;
		size_t column_index_;
		T* matrix_; // weak pointer to matrix_
	public:
		
		// this is the constructor normally used
		explicit iterator( size_t const dim_columns, T* const p ) : 
								dim_columns_(dim_columns), row_index_(0), column_index_(0), matrix_(p) { } 
		
		 // this can construct the end() iterator
		explicit iterator( size_t dim_rows_ ) : row_index_(dim_rows_), column_index_(0), matrix_(nullptr) { }
		
		iterator() : dim_columns_(0), row_index_(0), column_index_(0), matrix_(nullptr) { }
		~iterator() = default;
		
		// goes through the indices of matrix_
		// incrementing column_index_ at a time
		iterator& operator++() {
			++column_index_; 
			if (dim_columns_ <= column_index_) 
				++row_index_;
			column_index_ %= dim_columns_;
			return *this;
		}
		
		iterator operator++(int) {
			iterator temp{ *this };
			++(*this);
			return *this;
		}
		
		// two iterators are equivalent, when they point to the same element in matrix_
		bool operator==(iterator other) const {
			return ((row_index_ == other.row_index_) && (column_index_ == other.column_index_)); 
		}
		
		bool operator!=(iterator const& other) { return !((*this) == other); }
		T& operator*() const { return matrix_[ID(row_index_, column_index_, dim_columns_)]; }
	};

	iterator begin() { return iterator( dim_columns_, matrix_.get() ); }
	iterator end() { return iterator(dim_rows_); }

	// computes 
	// matrix[row_id] = ( constant * matrix[row_id] )
	void multiplyRow(size_t const& row_id, T const& constant){
		for(size_t j=0;j<dim_columns_;++j)
			matrix_[ID(row_id, j, dim_columns_)] *= constant;
	}

	// computes 
	// matrix[row_id_target] = matrix[row_id_target] + ( constant * matrix[row_id_source] )
	void affineTransform(size_t const& row_id_source, T const& constant, size_t const& row_id_target){
		for(size_t j=0;j<dim_columns_;++j)
			matrix_[ID(row_id_target, j, dim_columns_)] += ( constant * matrix_[ID(row_id_source, j, dim_columns_)] );
	}

	// get dim_rows_
	size_t get_dim_rows() const { 
		return dim_rows_;
	}

	// get dim_columns_
	size_t get_dim_columns() const { 
		return dim_columns_;
	}
};

// performs one step of the simplex algorithm.
// 'matrix' is the extended linear equation system with slack variables;
// moreover, the first row of 'matrix' is the linear function to be maximized.
template <class T>
bool simplexStep(Matrix<T>& matrix){
	T temp { matrix( 0, 0 ) };
	size_t col_index { 0 };
	size_t last_column { matrix.get_dim_columns() - 1 };
	
	// let's find the column of the pivot
	for(size_t j=1; j < last_column; ++j)
		if ( matrix( 0, j ) < temp ){
				temp = matrix( 0, j );
				col_index = j;
		}
	// std::cerr << "temp" << temp << std::endl;
	if ( temp >= 0 )
		return false; // the algorithm terminates
	
	size_t row_index { 1 };
	T fraction { matrix( row_index, last_column ) / matrix( row_index, col_index ) };
	
	// let's find the row of the pivot
	for(size_t j=2; j < matrix.get_dim_rows(); ++j){
		temp = matrix( j, last_column ) / matrix( j, col_index );
		if( temp < fraction ){
			fraction = temp;
			row_index = j;
		}
	}
	
	// the pivot is now 'matrix[row_index][column_index]'
	
	matrix.multiplyRow( row_index, static_cast<T>(1.0f) / matrix( row_index, col_index ) );
	// the pivot is now 'static_cast<T>(1.0)'
	
	for(size_t j=0; j < matrix.get_dim_rows(); ++j)
		if ( j != row_index )
			matrix.affineTransform( row_index, static_cast<T>(-1.0f) * matrix( j, col_index ), j );
	// now
	// 'matrix[j][column_index] == static_cast<T>(0.0)'
	// in each row, except for the row of the pivot
	
	return true; // simplex step was successful
}

// performs the simplex algorithm until either it terminates, or the maximum number of steps is reached;
// the maximum number of steps should be a function that's linear in the column dimension of 'matrix';;
// simplex is linear in expected computation time;
// and does not terminate in the worst case.
// -----------------------------------------
// -----------------------------------------
// parameters:
// ----------
// 'matrix' is the extended linear equation system with the slack variables; 
// Moreover, the first row of 'matrix' is the linear function to be *maximized*. 
// 'max_steps' is the upper bound on the number of simplex steps to be performed; 
template <class T>
void performSimplex(Matrix<T>& matrix, unsigned int const& max_steps ){
	unsigned int step { 0 };
	while( (step < max_steps) && simplexStep(matrix) )
		++step;
}

// Calls the above, with default value for max_steps being ( 2 * matrix.get_dim_columns() ). 
template <class T>
inline void performSimplex(Matrix<T>& matrix ){
	performSimplex<T> ( matrix, 2 * matrix.get_dim_columns() );
}

// to format the output stream into matrix form
// we will need 'my_ostream_iterator'
// the necessary typedefs and methods are being inherited
template <typename T>
class my_ostream_iterator : public std::ostream_iterator<T>{
	private:
		size_t dim_columns_;
		size_t column_index_;
		std::ostream& output_; // we need this, because std::ostream_iterator<T>::_M_stream is private
	public:
		explicit my_ostream_iterator( std::ostream& op, const char* delimiter, size_t d ) : 
										std::ostream_iterator<T>(op, delimiter), dim_columns_(d), column_index_(1), output_(op) { }
		my_ostream_iterator() = default;
		~my_ostream_iterator() = default;
		
		// we insert an 'std::endl' into the stream, 
		// before the beginning each row, 
		// except for the first one. 
		my_ostream_iterator& operator++() {
			column_index_ %= dim_columns_;
			++column_index_;
			if (column_index_ == 1)
				output_ << std::endl;
			std::ostream_iterator<T>::operator ++();
			return (*this);
		}
};

int main(){
	size_t const floating_point_precision_of_digits { 0 };
	
	// reading two numbers from file 'dimensions.txt'
	std::ifstream input_dimensions("dimensions.txt");
	std::istream_iterator<float> id_iterator(input_dimensions);
	size_t dim_rows { static_cast<size_t>(*id_iterator) };
	++id_iterator;
	size_t dim_columns { static_cast<size_t>(*id_iterator) };
	
	// Reading the input matrix from 'input.txt'. 
	// We assume the linear inequality constraints have already been turned into a linear 
	// equation system, by the introduction of slack variables. 
	// Moreover, we require that the first row contains the function to be *maximized*. 
	// i.e. we assume that the LP problem is in canonical form. 
	Matrix<float> matrix(dim_rows, dim_columns);
	std::ifstream input_matrix("input.txt");
	std::copy( std::istream_iterator<float>(input_matrix),
		   std::istream_iterator<float>(),
		   matrix.begin() );
	
	performSimplex( matrix ); // performs the simplex algorithm on matrix
							  // the first row of the matrix contains the 
							  // function to be *maximized*
	
	std::ofstream output_file("output.txt");
	output_file.precision( floating_point_precision_of_digits );
	output_file << std::fixed;
	
	// Writing into file with a non-buffered stream is slow. 
	// But it's going to be satisfactory, for our purposes here. 
	std::copy( matrix.begin(),
		   matrix.end(),
		   my_ostream_iterator<float>(output_file," ",dim_columns) ); // if you want it without the extra formatting, use: std::ostream_iterator<float>(output_file, " ") );
	output_file.flush();
  return 0;
}
