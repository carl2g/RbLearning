require_relative './lib/MatrixLib'
require "csv"

class Input

	attr_accessor :input_size

	def initialize(input_size)
		self.input_size = input_size
	end

end

class Matrix

	attr_accessor :matrix, :size_x, :size_y

	def initialize(size_y = 1, size_x = size_y, val = 0)
		self.size_x = size_x
		self.size_y = size_y
		self.matrix = Array.new(size_y * size_x) { |i| val }
	end

	def get2DArr
		Array.new(@size_y) { |y| Array.new(@size_x) { |x| self.matrix[y * @size_x + x] } }
	end

	def[](y, x = nil)
		if x.nil?
			getLines(y)
		else
			self.matrix[y * @size_x + x]
		end
	end

	def []=(y, x, val)
		self.matrix[y * @size_x + x] = val
	end

	def setLine(y, arr)
		(0...arr.size).each { |i| self[y, i] = arr[i] }
	end

	def set(arr)
		self.matrix = arr.flat_map do |line|
			line.map do |val|
				val.to_f
			end
		end
	end

	def getShape
		[self.size_y, self.size_x]
	end

	def printShape
		puts "size_y: #{self.size_y} size_x: #{self.size_x}"
	end

	def self.set(arr, size_y: arr.size, size_x: arr.first.size)
		Matrix.setVectorizedMatrix(arr.flatten, size_y, size_x)
	end

	def size_x
		@size_x
	end

	def size_y
		@size_y
	end

	def printM(round = 1)
		(0...@size_y).each do |y|
			(0...@size_x).each do |x|
				print "%.#{round}f  " % self[y, x]
			end
			puts
		end
	end

	def transpose
		vec = self.to_vect
		ptr = FFI::MemoryPointer.new(:double, vec.size)
		ptr.write_array_of_double(vec)
		res = MatrixLib.transpose(ptr, self.size_y, self.size_x)
		vect = res.read_array_of_double(self.size_y * self.size_x)
		m = Matrix.setVectorizedMatrix(vect, self.size_x, self.size_y)
		LibC.free(res)
		return m
	end

	def *(matrix)
		throw "[#{self.size_y}, #{self.size_x}] [#{matrix.size_y}, #{matrix.size_x}]" if self.size_x != matrix.size_y
		vec1 = self.to_vect
		ptr1 = FFI::MemoryPointer.new(:double, vec1.size)
		ptr1.write_array_of_double(vec1)
		vec2 = matrix.to_vect
		ptr2 = FFI::MemoryPointer.new(:double, vec2.size)
		ptr2.write_array_of_double(vec2)
		res = MatrixLib.dot(ptr1, ptr2, self.size_y, matrix.size_x, self.size_x)
		vect = res.read_array_of_double(self.size_y * matrix.size_x)
		m = Matrix.setVectorizedMatrix(vect, self.size_y, matrix.size_x)
		LibC.free(res)
		return m
	end

	def +(matrix)
		m = Matrix.boardcasting(matrix, self.size_y, self.size_x)
		vec1 = self.to_vect
		ptr1 = FFI::MemoryPointer.new(:double, vec1.size)
		ptr1.write_array_of_double(vec1)
		vec2 = m.to_vect
		ptr2 = FFI::MemoryPointer.new(:double, vec2.size)
		ptr2.write_array_of_double(vec2)
		res = MatrixLib.add(ptr1, ptr2, self.size_y * self.size_x)
		vect = res.read_array_of_double(self.size_y * self.size_x)
		m = Matrix.setVectorizedMatrix(vect, self.size_y, self.size_x)
		LibC.free(res)
		return m
	end

	def ^(n)
		return Matrix.setVectorizedMatrix(
				self.matrix.map { |v| v**n },
				self.size_y,
				self.size_x
			)
	end

	def **(matrix)
		m = Matrix.boardcasting(matrix, self.size_y, self.size_x)
		vec1 = self.to_vect
		ptr1 = FFI::MemoryPointer.new(:double, vec1.size)
		ptr1.write_array_of_double(vec1)
		vec2 = m.to_vect
		ptr2 = FFI::MemoryPointer.new(:double, vec2.size)
		ptr2.write_array_of_double(vec2)
		res = MatrixLib.mult(ptr1, ptr2, self.size_y * self.size_x)
		vect = res.read_array_of_double(self.size_y * self.size_x)
		m = Matrix.setVectorizedMatrix(vect, self.size_y, self.size_x)
		LibC.free(res)
		return m
	end

	def -(matrix)
		m = Matrix.boardcasting(matrix, self.size_y, self.size_x)
		vec1 = self.to_vect
		ptr1 = FFI::MemoryPointer.new(:double, vec1.size)
		ptr1.write_array_of_double(vec1)
		vec2 = m.to_vect
		ptr2 = FFI::MemoryPointer.new(:double, vec2.size)
		ptr2.write_array_of_double(vec2)
		res = MatrixLib.subtract(ptr1, ptr2, self.size_y * self.size_x)
		vect = res.read_array_of_double(self.size_y * self.size_x)
		m = Matrix.setVectorizedMatrix(vect, self.size_y, self.size_x)
		LibC.free(res)
		return m
	end

	def <<(matrix)
		newM = Matrix.new(self.size_y, self.size_x + matrix.size_x)
		(0...self.size_y).each do |y|
			(0...self.size_x).each do |x|
				newM[y, x] = self[y, x]
			end
		end
		(0...matrix.size_y).each do |y|
			(0...matrix.size_x).each do |x|
				newM[y, self.size_x + x] = matrix[y, x]
			end
		end
		return newM
	end

	def >>(matrix)
		newM = Matrix.new(self.size_y, self.size_x + matrix.size_x)
		(0...self.size_y).each do |y|
			(0...self.size_x).each do |x|
				newM[y, x] = matrix[y, x]
			end
		end
		(0...matrix.size_y).each do |y|
			(0...matrix.size_x).each do |x|
				newM[y, self.size_x + x] = self[y, x]
			end
		end
		return newM
	end

	def dump_matrix(mat)
		self.matrix = mat.matrix.clone
		self.reshape(mat.getShape)
	end

	def drop_row(i)
		first_part_mat = self.getMat(0, 0, i, self.size_x)
		sec_part_mat = self.getMat(i + 1, 0, self.size_y - (i + 1) , self.size_x)
		self.dump_matrix((first_part_mat.transpose << sec_part_mat.transpose).transpose)
	end

	def to_vect
		self.matrix
	end

	def self.setVectorizedMatrix(vect, size_y, size_x)
		m = Matrix.new(size_y, size_x)
		m.matrix = vect.map { |val| val.to_f }
		return m
	end

	def copy
		Matrix.setVectorizedMatrix(self.to_vect, self.size_y, self.size_x)
	end

	def round(n = 1)
		self.matrix = self.matrix.map do |v|
			v.round(n)
		end
	end

	def applyOp(op, nb)
		Matrix.setVectorizedMatrix(self.matrix.each_with_index.map do |val, i|
			val.send(op, nb)
		end, self.size_y, self.size_x)
	end

	def set_if(set = 0, nb = 0, op = :<)
		self.matrix.each_with_index do |val, i|
			self[i] = set if val.send(op, nb)
		end
	end

	def sum(beg_y = 0, beg_x = 0, size_y = self.size_y, size_x = self.size_x)
		sum = 0
		(beg_y...(beg_y + size_y)).each do |y|
			(beg_x...(beg_x + size_x)).each do |x|
				sum += self[y, x] if self[y] && self[y, x]
			end
		end
		return sum.to_f
	end

	def sumAxis(start: 0, finish: self.size_x)
		Matrix.set(
		      (0...self.size_y).map do |y|
		           	[(start...finish).sum do |x|
		         		self[y, x]
		           	end]
		      end
		)
	end

	def sumOrd
		Matrix.set(
		      [(0...self.size_x).map do |x|
		      	(0...self.size_y).sum do |y|
		         		self[y, x]
		           	end
		      end]
		)
	end

	def split(size_y, size_x, step_y = size_y, step_x = size_x)
		arr = []
		(0...self.size_y).step(step_y) do |y|
			(0...self.size_x).step(step_x) do |x|
				arr += [self.getMat(y, x, size_y, size_x)]
			end
		end
		return arr
	end

	def to_csv
		csv = CSV.generate do |csv|
			(0...self.size_y).each do |y|
				csv << self.getLines(y)
			end
		end
		return csv
	end

	def getMat(beg_y, beg_x, size_y, size_x)
		newM = Matrix.new(size_y, size_x)
		(0...size_y).each do |y|
			(0...size_x).each do |x|
				newM[y, x] = self[beg_y + y, beg_x + x] if self[beg_y + y] && self[beg_y + y, beg_x + x]
			end
		end
		return newM
	end

	def getMax(beg_y = 0, beg_x = 0, size_y = self.size_y, size_x = self.size_x)
		max = self[beg_y, beg_x]
		(beg_y...beg_y + size_y).each do |y|
			(beg_x...beg_x + size_x).each do |x|
				max = max < self[y, x] ? self[y, x] : max
			end
		end
		return max
	end

	def self.boardcasting(matrix, size_y, size_x)
		Matrix.set(
		      (0...size_y).map do |y|
		      	(0...size_x).map do |x|
		      		matrix[y % matrix.size_y, x % matrix.size_x]
		      	end
		      end
		)
	end

	def min
		self.matrix.min
	end

	def max
		self.matrix.max
	end

	def reshape(shape)
		self.size_y = shape[0]
		self.size_x = shape[1]
	end

	def getLines(y)
		range = (y * @size_x)...(y * @size_x + @size_x)
		return self.matrix[range]
	end

end
