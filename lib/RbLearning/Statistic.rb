module Statistic

	def mean(arr)
		return arr.sum { |val| val.to_f } / arr.size
	end

	def quartile(datas, q)
		datas.map!(&:to_f)
		datas.sort[(datas.size * q) / 4]
	end

	def esperance(arr)
		e = 0
		vals = arr.uniq
		vals.each do |v|
			e += (arr.count(v).to_f / arr.size) * v.to_f
		end
		return e
	end

	def cov(arr1, arr2)
		mean1 = mean(arr1)
		mean2 = mean(arr2)

		tmp = (0...arr1.size).sum do |i|
			(arr1[i].to_f - mean1) * (arr2[i].to_f - mean2)
		end
		return tmp / (arr1.size - 1)
	end

	def skewness(arr)
		mean = mean(arr)
		std_dev = std_dev(arr)
		skew = 0
		arr.each do |val|
			skew += ((val.to_f - mean) / std_dev)**3
		end
		return skew / arr.size
	end

	def correlation(arr1, arr2)
		cov(arr1, arr2) / (std_dev(arr1) * std_dev(arr2))
	end

	def variance(arr)
		dev = 0
		mean = self.mean(arr)
		var = arr.sum { |val| (mean - val.to_f)**2 }
		return var / (arr.size - 1)
	end

	def std_dev(arr)
		Math.sqrt(variance(arr))
	end

	def normalize_range(arr)
		min = arr.min
		max = arr.max
		delt = max - min
		(0...arr.size).each do |i|
			arr[i] = (arr[i] - min) / delt
		end
		return (arr)
	end

	def mat_normalize_range(mat, axis: 0)
		m = axis == 1 ? mat.transpose : mat
		
		(0...m.size_y).each do |i|
			m.setLine(i, normalize_range(m[i]))
		end

		return axis == 1 ? m.transpose : m
	end

	def standerdize(arr, mean: mean(arr), std_dev: std_dev(arr) )
		(0...arr.size).each do |i|
			arr[i] = (arr[i] - mean) / std_dev
		end
		return arr
	end

	def mat_standerdize(mat, axis: 0)
		m = axis == 1 ? mat.transpose : mat
		
		(0...m.size_y).each do |i|
			m.setLine(i, standerdize(m[i]))
		end

		return axis == 1 ? m.transpose : m
	end

	# def standerdized(mat)
	# 	mean = mean(mat.matrix)		
	# 	std  = std_dev(mat.matrix)

	# 	mat.matrix = mat.matrix.map do |v|
	# 		(v - mean) / std
	# 	end

	# 	return mat
	# end

	def relative_frequence(arr, val)
		return frequence(arr, val) / arr.size.to_f
	end

	def frequence(arr, val)
		return arr.count(val)
	end

end
