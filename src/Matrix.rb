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

	def set(arr)
		@matrix = arr.flat_map do |line|
			line.map do |val|
				val.to_f
			end
		end
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
		puts
	end

	def transpose
		vec = self.to_vect
		ptr = FFI::MemoryPointer.new(:float, vec.size)
		ptr.write_array_of_float(vec)
		res = MatrixLib.transpose(ptr, self.size_y, self.size_x)
		vect = res.read_array_of_float(self.size_y * self.size_x)
		m = Matrix.setVectorizedMatrix(vect, self.size_x, self.size_y)
		LibC.free(res)
		return m
	end

	def *(matrix)
		vec1 = self.to_vect
		ptr1 = FFI::MemoryPointer.new(:float, vec1.size)
		ptr1.write_array_of_float(vec1)
		vec2 = matrix.to_vect
		ptr2 = FFI::MemoryPointer.new(:float, vec2.size)
		ptr2.write_array_of_float(vec2)
		res = MatrixLib.dot(ptr1, ptr2, self.size_y, matrix.size_x, self.size_x)
		vect = res.read_array_of_float(self.size_y * matrix.size_x)
		m = Matrix.setVectorizedMatrix(vect, self.size_y, matrix.size_x)
		LibC.free(res)
		return m
	end

	def +(matrix)
		m = boardcasting(matrix, self.size_y, self.size_x)
		vec1 = self.to_vect
		ptr1 = FFI::MemoryPointer.new(:float, vec1.size)
		ptr1.write_array_of_float(vec1)
		vec2 = m.to_vect
		ptr2 = FFI::MemoryPointer.new(:float, vec2.size)
		ptr2.write_array_of_float(vec2)
		res = MatrixLib.add(ptr1, ptr2, self.size_y, self.size_x)
		vect = res.read_array_of_float(self.size_y * self.size_x)
		m = Matrix.setVectorizedMatrix(vect, self.size_y, self.size_x)
		LibC.free(res)
		return m
	end

	def **(matrix)
		m = boardcasting(matrix, self.size_y, self.size_x)
		vec1 = self.to_vect
		ptr1 = FFI::MemoryPointer.new(:float, vec1.size)
		ptr1.write_array_of_float(vec1)
		vec2 = m.to_vect
		ptr2 = FFI::MemoryPointer.new(:float, vec2.size)
		ptr2.write_array_of_float(vec2)
		res = MatrixLib.mult(ptr1, ptr2, self.size_y, self.size_x)
		vect = res.read_array_of_float(self.size_y * self.size_x)
		m = Matrix.setVectorizedMatrix(vect, self.size_y, self.size_x)
		LibC.free(res)
		return m
	end

	def -(matrix)
		m = boardcasting(matrix, self.size_y, self.size_x)
		vec1 = self.to_vect
		ptr1 = FFI::MemoryPointer.new(:float, vec1.size)
		ptr1.write_array_of_float(vec1)
		vec2 = m.to_vect
		ptr2 = FFI::MemoryPointer.new(:float, vec2.size)
		ptr2.write_array_of_float(vec2)
		res = MatrixLib.subtract(ptr1, ptr2, self.size_y, self.size_x)
		vect = res.read_array_of_float(self.size_y * self.size_x)
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
		Matrix.setVectorizedMatrix(self.to_vect)
	end

	def filter(filter, pos_y = 0, pos_x = 0, op = :* )
		newM = self.copy
		filter.matrix.each_with_index do |line, y|
			line.each_with_index do |val, x|
				newM[pos_y + y][pos_x + x] = newM[pos_y + y][pos_x + x].send(op, val) if newM[pos_y + y] && newM[pos_y + y][pos_x + x]
			end
		end
		return newM
	end

	def self.filterAll(m, filter, op = :*)
		filtered_m = Matrix.new(m.size_y, m.size_x)
		(0...m.size_y).each do |y|
			(0...m.size_x).each do |x|
				tmp = m.filter(filter, y, x, op)
				sum = tmp.sum(y, x, filter.size_y, filter.size_x)
				filtered_m[y, x] = sum
			end
		end
		return filtered_m
	end

	def applyOp(op, nb)
		self.matrix.each_with_index do |val, i|
			@matrix[i] = val.send(op, nb)
		end
		return self
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
		      	tmp = 0
		           	(0...self.size_x).each do |x|
		         		tmp += self[y, x]
		           	end
		           	[tmp]
		      end
		)
	end

	def sumOrd
		Matrix.set(
		      (0...self.size_x).map do |x|
		      	tmp = 0
		      	(0...self.size_y).each do |y|
		         		tmp += self[y, x]
		           	end
		           	[tmp]
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
		max = self[0][0]
		(beg_y...beg_y + size_y).each do |y|
			(beg_x..beg_x + size_x).each do |x|
				max = self[y, x] if self[y] && self[y, x] && self[y, x] > max
			end
		end
		return max
	end

	def normalize(norm)
		self.matrix.each_with_index do |val, i|
			@matrix[i] = val / norm
		end
	end

private

	def boardcasting(matrix, size_y, size_x)
		Matrix.set(
		      (0...size_y).map do |y|
		      	(0...size_x).map do |x|
		      		matrix[y % matrix.size_y, x % matrix.size_x]
		      	end
		      end
		)
	end

	def getLines(y)
		@matrix[(y.first * size_x)...(y.last * size_x)]
	end

end
