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
		      	x > 0 ? 1.0 : 0.0
			end,
			m.size_y,
			m.size_x
		)
	end

	def sigmoidUnit(x)
		1.0 / (1.0 + CMath.c_exp(-x))
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
		      	CMath.c_exp(-x) / (1.0 + CMath.c_exp(-x))**2
			end,
			m.size_y,
			m.size_x
		)
	end

	def tanh(m)
		Matrix.setVectorizedMatrix(
		      m.matrix.map do |x|
				CMath.c_tanh(x)
			end,
			m.size_y,
			m.size_x
		)
	end

	def tanhPrime(m)
		Matrix.setVectorizedMatrix(
		      m.matrix.map do |x|
				1.0 - Math.tanh(x)**2
			end,
			m.size_y,
			m.size_x
		)
	end

	def softMax(m)
		Matrix.set((0...m.size_y).map do |y|
			sum = 0.0
			out = (0...m.size_x).map do |x|
				tmp = CMath.c_exp(m[y, x])
				sum += tmp
				tmp
			end
			out.map { |val| (val / sum) }
		end)
	end

end
