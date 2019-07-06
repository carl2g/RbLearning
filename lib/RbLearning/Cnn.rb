require_relative './ActivFunc'
require_relative './LossFunc'
require_relative './NeuroNet'

class Cnn < NeuroNet

	attr_accessor :ann, :filters

	def initilize
		@ann = NeuroNet.new
	end

	# def addFilter(size_y, size_x, min = -1.0, max = 1.0)
	# 	mat = Matrix.new(size_y, size_x)
	# 	r = Random.new
	# 	(0...size_y).each do |y|
	# 		(0...size_x).each do |x|
	# 			mat[y][x] = r.rand(min..max)
	# 		end
	# 	end
	# 	@filters << mat
	# 	return @filters
	# end

	# def applyConv2D(data)
	# 	filtered_matrix = []
	# 	@filters.each do |filter|
	# 		filtered_matrix << self.filter(data, filter)
	# 	end
	# 	return filtered_matrix
	# end

	def convolution(data, filter, step_y: 1, step_x: 1)
		filtered_m = Matrix.set(
			(0...data.size_y).step(step_y).map do |y|
				(0...data.size_x).step(step_x).map do |x|
					applyFilter(data, filter, y, x)
				end
			end
		)
		return filtered_m
	end

	def maxPool(matrix, step_y: 2, step_x: 2, pool_size_y: step_y, pool_size_x: step_x)
		pooled_mat = Matrix.set(
		      (0...matrix.size_y).step(step_y).map do |y|
			      (0...matrix.size_x).step(step_x).map do |x|
					matrix.getMat(y, x, pool_size_y, pool_size_x).getMax
				end
		      end
		)
		return pooled_mat
	end

	def maxpoolBackward(pooled_data: nil, original_data: nil, step_y: 2, step_x: 2, pool_size_y: step_y, pool_size_x: step_x)
		m = Matrix.new(orig.size_y, orig.size_x)
		(0...orig.size_y).each do |l|
			(0...orig.size_x).each do |v|
				(l...l + step_y).each do |y|
					(v...v + step_x).each do |x|
						m[y][x] = orig.getMax(l, v, l + pool_size_y, v + pool_size_x) if m[y] && m[y][x]
					end
				end
			end
		end
		return m
	end

	# def findInM(image, mat)
	# 	min = -1
	# 	(0..(image.size_y - mat.size_y)).each do |y|
	# 		(0..(image.size_x - mat.size_x)).each do |x|
	# 			tmp = image.filter(mat, y, x, :-)
	# 			(0...tmp.size_y).each { |y| (0...tmp.size_x).each { |x| tmp[y][x] = tmp[y][x].abs } }
	# 			sum = tmp.sum(y, x, mat.size_y, mat.size_x)
	# 			min = sum if min > sum || min < 0
	# 		end
	# 	end
	# 	return min
	# end

	private

	def applyFilter(img, filter, pos_y = 0, pos_x = 0 )
		newM = Matrix.new(filter.size_y, filter.size_x)
		(0...filter.size_y).each do |y|
			(0...filter.size_x).each do |x|
				newM[y, x] = img[pos_y + y, pos_x + x] * filter[y, x] if img[pos_y + y, pos_x + x]
			end
		end
		return newM.sum
	end
end
