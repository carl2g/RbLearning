module ActivFunc

	def reLu(m)
		Matrix.setVectorizedMatrix(
		      m.matrix.map do |x|
		      	x > 0 ? x : 0
			end,
			m.size_y,
			m.size_x
		)
	end

	def reLuPrime(m)
		Matrix.setVectorizedMatrix(
		      m.matrix.map do |x|
		      	x > 0 ? 1 : 0.0
			end,
			m.size_y,
			m.size_x
		)
	end

	def sigmoidUnit(x)
		1.0 / (1.0 + CMath.expo(-x))
	end

	def sigmoid(m)
		Matrix.setVectorizedMatrix(
		      m.matrix.map do |x|
				sigmoidUnit(x)
			end,
			m.size_y,
			m.size_x
		)
	end

	def sigmoidPrime(m)
		Matrix.setVectorizedMatrix(
		      m.matrix.map do |x|
		      	CMath.expo(-x) / (1.0 + CMath.expo(-x))**2
			end,
			m.size_y,
			m.size_x
		)
	end

	def tanh(m)
		Matrix.setVectorizedMatrix(
		      m.matrix.map do |x|
				(CMath.expo(x) - CMath.expo(-x)) / (CMath.expo(x) + CMath.expo(-x))
			end,
			m.size_y,
			m.size_x
		)
	end

	def tanhPrime(m)
		Matrix.setVectorizedMatrix(
		      m.matrix.map do |x|
				1.0 - CMath.tanh(x)**2
			end,
			m.size_y,
			m.size_x
		)
	end

	def softMax(vect)
		sum = 0.0
		out = vect.map do |val|
			tmp = CMath.expo(val)
			sum += tmp
			tmp
		end
		out = out.map {|val| (val / sum) }
		return out
	end

end
