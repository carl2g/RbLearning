module ActivFunc

	def reLu(m)
		Matrix.set(
		      m.matrix.map do |vect|
				vect.map { |x| x > 0 ? x : 0 }
			end
		)
	end

	def sigmoid(m)
		Matrix.set(
		      m.matrix.map do |vect|
				vect.map { |x| 1.0 / (1.0 + Math.exp(-x)) }
			end
		)
	end

	def sigmoidPrime(m)
		Matrix.set(
		      m.matrix.map do |vect|
		      	vect.map { |x| x * (1 - x) }
			end
		)
	end

	def tanh(vect)
		vect.map { |x| 2.0 / (1.0 + Math.exp(-2 * x)) - 1 }
	end

	def dtanh(vect)
		vect.map { |x| 1 - Math.tanh(x)**2 }
	end

	def softMax(vect)
		sum = 0.0
		out = vect.map do |val|
			tmp = Math.exp(val)
			sum += tmp
			tmp
		end
		out = out.map {|val| (val / sum) }
		return out
	end

end
