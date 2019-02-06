require_relative './ActivFunc'

class Cnn

	include ActivFunc

	attr_accessor :hashed_data, :size_x, :size_y, :matrix

	def initialize(csv_file)
		content 		= CSV.read(csv_file)
		@hashed_data 	= {}
		content.shift.each { |key| @hashed_data[key] = [] }
		@hashed_data.each_with_index do |h, i|
			key = h.first
			@hashed_data[key] += content.map do |line|
				line[i]
			end
		end
		@size_x = @hashed_data.size
		@size_y = @hashed_data.first.last.size
	end

	def addFilter(size_y, size_x, min = -1.0, max = 1.0)
		mat = Matrix.new(size_y, size_x)
		(0...size_y).each do |y|
			(0...size_x).each do |x|
				r = Random.new
				mat[y, x] = r.rand(min..max)
			end
		end
		return mat
	end

	def initFilter(size_y, size_x, nb)
		filters = []
		(0...nb).each do |i|
			filters += [addFilter(size_y, size_x)]
		end
		return filters
	end

	def filter(x_train, filter)
		mat = Matrix.new
		mat.set(
			(0...x_train.size_y).map do |i|
				m = Matrix.convertToMatrix(x_train[i])
				filtered_m = Matrix.filterAll(m, filter, :*)
				filtered_m.to_vect
			end
		)
		return mat
	end

	def normalize(mat, norm)
		mat.normalize(norm)
		return mat
	end

	def max_pooling(matrix, step_y = 4, step_x = 4, size_y = Math.sqrt(matrix.size_x), size_x = Math.sqrt(matrix.size_x))
		pooled_mat = Matrix.new
		pooled_mat.set(matrix.matrix.map do |vect|
			mat = Matrix.convertToMatrix(vect)
			mat.set(
			        (0...mat.size_y).step(step_y).map do |y|
					(0...mat.size_y).step(step_x).map do |x|
						mat.getMat(y, x, step_y, step_x).getMax
					end
				end
			)
			mat.to_vect
		end )
		return pooled_mat
	end

	def maxpoolBackward(pooled, orig, filter, step)
		m = Matrix.new(orig.size_y, orig.size_x)
		(0...orig.size_y).each do |l|
			(0...orig.size_x).each do |v|
				(l...l + step).each do |y|
					(v...v + step).each do |x|
						m[y, x] = orig.getMax(l, v, filter.size_y, filter.size_x) if m[y] && m[y, x]
					end
				end
			end
		end
		return m
	end

	def findInM(image, mat)
		min = -1
		(0..(image.size_y - mat.size_y)).each do |y|
			(0..(image.size_x - mat.size_x)).each do |x|
				tmp = image.filter(mat, y, x, :-)
				(0...tmp.size_y).each { |y| (0...tmp.size_x).each { |x| tmp[y, x] = tmp[y, x].abs } }
				sum = tmp.sum(y, x, mat.size_y, mat.size_x)
				min = sum if min > sum || min < 0
			end
		end
		return min
	end

	def matrixGen(with = self.labels)
		newH = @hashed_data.select { |key, val| with.include?(key) }
		size_y = @size_y
		size_x = newH.keys.size
		arr = Matrix.new(size_y, size_x)
		(0...size_y).each do |x|
			newH.each_with_index do |h, y|
				key, val = h
				arr[x, y] = val[x].to_f
			end
		end
		@matrix = arr
	end

	def matrix
		@matrix
	end

	def remove(label)
		@hashed_data.delete(label)
	end

	def [](key)
		@hashed_data[key]
	end

	def []=(key, values)
		@hashed_data[key] = values
	end

	def labels
		@hashed_data.keys
	end

end
