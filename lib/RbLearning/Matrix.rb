require_relative './lib/MatrixLib'

class Matrix

	attr_accessor :matrix, :size_x, :size_y

	def initialize(size_y = 0, size_x = size_y, val = 0)
		@size_x = size_x
		@size_y = size_y
		@matrix = Array.new(@size_y *  @size_x) { |i| val }
	end

	def get2DArr
		Array.new(@size_y) { |y| Array.new(@size_x) { |x| @matrix[y * @size_x + x] } }
	end

	def[](y, x = nil)
		if x.nil?
			getLines(y)
		else
			@matrix[y * @size_x + x]
		end
	end

	def []=(y, x, val)
		@matrix[y * @size_x + x] = val
	end

	def setLine(y, arr)
		(0...arr.size).each { |i| @matrix[y, i] = arr[i] }
	end

	def set(arr)
		@matrix = arr.flat_map do |line|
			line.map do |val|
				val.to_f
			end
		end
	end

	def dimensions
		[self.size_y, self.size_x]
	end

	def printShape
		puts "size_y: #{self.size_y} size_x: #{self.size_x}"
	end

	def self.set(arr)
		size_y = arr.size
		size_x = arr.first.size
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

	def sumAxis
		Matrix.set(
		      (0...self.size_y).map do |y|
		           	[(0...self.size_x).sum do |x|
		         		self[y, x]
		           	end]
		      end
		)
	end

	def sumOrd
		Matrix.set(
		      (0...self.size_x).map do |x|
		      	[(0...self.size_y).sum do |y|
		         		self[y, x]
		           	end]
		      end
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

	def getMat(beg_y, beg_x, size_y, size_x)
		newM = Matrix.new(size_y, size_x)
		(0...size_y).each do |y|
			(0...size_x).each do |x|
				newM[y, x] = self[beg_y + y][beg_x + x] if self[beg_y + y] && self[beg_y + y][beg_x + x]
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

	def normalize(axis: nil)
		m = axis == 1 ? self.transpose : self
		
		if axis
			(0...m.size_y).each do |i|
				min = m[i].min
				max = m[i].max
				delt = (max - min) == 0 ? 1 : (max - min)
				(0...m[i].size).each do |x|
					m[i, x] = (m[i, x] - min) / delt
				end
			end
		else
			# std_dev = std_dev(m.matrix)
			# mean = mean(m.matrix)
			min = m.matrix.min
			max = m.matrix.max
			delt = (max - min) == 0 ? 1 : (max - min)
			m.matrix.each_with_index do |val, i|
				m.matrix[i] = (val - min) / delt
				# m.matrix[i] = (val - mean) / std_dev
			end
		end
		return axis == 1 ? m.transpose : m
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

	def getLines(y)
		if y.is_a? Range
			@matrix[(y.first * @size_x)...(y.last * @size_x)]
		else
			@matrix[(y * @size_x)...(y * @size_x + @size_x)]
		end
	end

end
