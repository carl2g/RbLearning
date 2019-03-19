module Statistics

	def mean(arr)
		res = 0.0
		arr.each { |val| res += val.to_f }
		return res / arr.size
	end

	def quartile(datas, q)
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
		tmp = 0
		(0...arr1.size).each do |i|
			tmp += (arr1[i].to_f - mean1) * (arr2[i].to_f - mean2)
		end
		return tmp / arr1.size
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

	def corelation(arr1, arr2)
		cov(arr1, arr2) / (std_dev(arr1) * std_dev(arr2))
	end

	def variance(arr)
		dev = 0
		mean = self.mean(arr)
		arr.each { |val| dev += (mean - val.to_f)**2 }
		return dev / arr.size.to_f
	end

	def std_dev(arr)
		Math.sqrt(variance(arr))
	end

end
