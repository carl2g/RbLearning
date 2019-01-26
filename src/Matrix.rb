class Matrix

	attr_accessor :matrix, :size_x, :size_y

	def initialize(size_y = 0, size_x = size_y, val = 0)
		@size_x = size_x
		@size_y = size_y
		@matrix = Array.new(@size_y) { |i| Array.new(@size_x) { |i| val }  }
	end

	def[](x)
		@matrix[x]
	end

	def []=(x, val)
		@matrix[x] = val
	end

	def set(arr)
		@size_x = arr.first.size
		@size_y = arr.size

		@matrix = arr.map do |line|
			line.map do |val|
				val.to_f
			end
		end
	end

	def printShape
		puts "size_y: #{self.size_y} size_x: #{self.size_x}"
	end

	def self.set(arr)
		m = Matrix.new

		m.set( arr.map do |line|
			line.map do |val|
				val.to_f
			end
		end)
		return m
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
				print "%.#{round}f  " % @matrix[y][x]
			end
			puts
		end
		puts
	end

	def transpose
		transposeM = Matrix.new(self.size_x, self.size_y)
		(0...transposeM.size_y).each do |y|
			(0...transposeM.size_x).each do |x|
				transposeM[y][x] = self[x][y]
			end
		end
		return transposeM
	end

	def *(matrix)
		newM = Matrix.new
		newM.set(
			(0...self.size_y).map do |y|
				(0...matrix.size_x).map do |l|
					val = 0
					(0...self.size_x).map do |x|
						val += self[y][x] * matrix[x][l]
					end
					val
				end
			end
		)
		return newM
	end

	def +(matrix)
		newM = Matrix.new(self.size_y, self.size_x)
		m = boardcasting(matrix, self.size_y, self.size_x)
		(0...self.size_y).each do |y|
			(0...self.size_x).each do |x|
				newM[y][x] = self[y][x] + m[y][x]
			end
		end
		return newM
	end

	def **(matrix)
		newM = Matrix.new(self.size_y, self.size_x)
		m = boardcasting(matrix, self.size_y, self.size_x)
		(0...self.size_y).each do |y|
			(0...self.size_x).each do |x|
				newM[y][x] = self[y][x] * m[y][x]
			end
		end
		return newM
	end

	def -(matrix)
		newM = Matrix.new(self.size_y, self.size_x)
		m = boardcasting(matrix, self.size_y, self.size_x)
		(0...self.size_y).each do |y|
			(0...self.size_x).each do |x|
				newM[y][x] = self[y][x] - m[y][x]
			end
		end
		return newM
	end

	def <<(matrix)
		newM = Matrix.new(self.size_y, self.size_x + matrix.size_x)
		(0...self.size_y).each do |y|
			(0...self.size_x).each do |x|
				newM[y][x] = self[y][x]
			end
		end
		(0...matrix.size_y).each do |y|
			(0...matrix.size_x).each do |x|
				newM[y][self.size_x + x] = matrix[y][x]
			end
		end
		return newM
	end

	def to_vect
		self.matrix.flatten
	end

	def self.convertToMatrix(vect, size_y = Math.sqrt(vect.size), size_x = size_y)
		newM = Matrix.new
		newM.set((0...size_y).map do |y|
			vect[(y * size_x)...(y * size_x + size_x)]
		end)
		return newM
	end

	def copy
		newM = Matrix.new
		newM.set(self.matrix)
		return newM
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
				filtered_m[y][x] = sum
			end
		end
		return filtered_m
	end

	def applyOp(op, nb)
		self.matrix.each_with_index do |line, y|
			line.each_with_index do |val, x|
				self[y][x] = val.send(op, nb)
			end
		end
		return self
	end

	def set_if(set = 0, nb = 0, op = :<)
		self.matrix.each_with_index do |line, y|
			line.each_with_index do |val, x|
				self[y][x] = set if val.send(op, nb)
			end
		end
	end

	def sum(beg_y = 0, beg_x = 0, size_y = self.size_y, size_x = self.size_x)
		sum = 0
		(beg_y...(beg_y + size_y)).each do |y|
			(beg_x...(beg_x + size_x)).each do |x|
				sum += self[y][x] if self[y] && self[y][x]
			end
		end
		return sum.to_f
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
				newM[y][x] = self[beg_y + y][beg_x + x] if self[beg_y + y] && self[beg_y + y][beg_x + x]
			end
		end
		return newM
	end

	def getMax(beg_y = 0, beg_x = 0, size_y = self.size_y, size_x = self.size_x)
		max = self[0][0]
		(beg_y...beg_y + size_y).each do |y|
			(beg_x..beg_x + size_x).each do |x|
				max = self[y][x] if self[y] && self[y][x] && self[y][x] > max
			end
		end
		return max
	end

	def normalize(norm)
		self.matrix.each_with_index do |line, y|
			line.each_with_index do |val, x|
				self[y][x] = val / norm
			end
		end
	end

private

	def boardcasting(matrix, size_y, size_x)
		Matrix.set(
		      (0...size_y).map do |y|
		      	(0...size_x).map do |x|
		      		matrix[y % matrix.size_y][x % matrix.size_x]
		      	end
		      end
		)
	end

end
