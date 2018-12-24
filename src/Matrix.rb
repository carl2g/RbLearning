class Matrix

	attr_accessor :matrix, :size_x, :size_y

	def initialize(size_y = 0, size_x = size_y)
		@size_x = size_x
		@size_y = size_y
		@matrix = Array.new(@size_y) { |i| Array.new(@size_x) { |i| 0.0 }  }
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

	def size_x
		@size_x
	end

	def size_y
		@size_y
	end

	def printM(round = 1)
		(0...@size_y).each do |y|
			(0...@size_x).each do |x|
				print "#{@matrix[y][x].round(round)}  "
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
		min_x = (self.size_x > matrix.size_x) ? matrix.size_x : self.size_x
		min_y = (self.size_y > matrix.size_y) ? matrix.size_y : self.size_y
		newM = Matrix.new(min_y, min_x)

		(0...min_x).each do |x|
			(0...min_y).each do |v|
				(0...self.size_y).each do |y|
					newM[v][x] += self[y][x] * matrix[v][y]
				end
			end
		end
		return newM
	end

	def +(matrix)
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

end
