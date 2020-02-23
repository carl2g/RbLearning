module ActivFunc

	class ReLu
		def self.func(m)
			Matrix.setVectorizedMatrix(
			      m.matrix.map do |x|
			      	x > 0 ? x : 0
				end,
				m.size_y,
				m.size_x
			)
		end 

		def self.derivate(m)
			Matrix.setVectorizedMatrix(
			      m.matrix.map do |x|
			      	x > 0 ? 1.0 : 0.0
				end,
				m.size_y,
				m.size_x
			)
		end
	end

	class LeakyReLu
		def self.func(m)
			Matrix.setVectorizedMatrix(
			      m.matrix.map do |x|
			      	x > 0 ? x : Math.exp(x) - 1
				end,
				m.size_y,
				m.size_x
			)
		end 

		def self.derivate(m)
			Matrix.setVectorizedMatrix(
			      m.matrix.map do |x|
			      	x > 0 ? 1.0 : Math.exp(x)
				end,
				m.size_y,
				m.size_x
			)
		end
	end

	class Sigmoid
		def self.sigmoidUnit(x)
			1.0 / (1.0 + CMath.exp(-x))
		end

		def self.func(m)
			Matrix.setVectorizedMatrix(
			    m.matrix.map do |x|
					sigmoidUnit(x)
				end,
				m.size_y,
				m.size_x
			)
		end

		def self.derivate(m)
			Matrix.setVectorizedMatrix(
			    m.matrix.map do |x|
			      	sigmoidUnit(x) * (1.0 - sigmoidUnit(x))
				end,
				m.size_y,
				m.size_x
			)
		end
	end

	class Tanh
		def self.func(m)
			Matrix.setVectorizedMatrix(
			      m.matrix.map do |x|
					CMath.tanh(x)
				end,
				m.size_y,
				m.size_x
			)
		end

		def self.derivate(m)
			Matrix.setVectorizedMatrix(
			      m.matrix.map do |x|
					1.0 - CMath.tanh(x)**2
				end,
				m.size_y,
				m.size_x
			)
		end
	end

	class SoftMax
		def self.func(m)
			m = Matrix.set((0...m.size_x).map do |x|
				sum = 0.0
				max = m.getMax(0, x, m.size_y, 1)
				out = (0...m.size_y).map do |y|

					tmp = CMath.exp(m[y, x].abs - max)
					sum += tmp
					tmp
					
				end
				out.map { |val| val / sum }
			end)
			return m.transpose
		end

		def self.derivate(m)
			m = Matrix.set((0...m.size_x).map do |x|
				
				max = m.getMax(0, x, m.size_y, 1)
				
				sum = (0...m.size_y).sum do |y|
					CMath.exp(m[y, x] - max)
				end

				(0...m.size_y).map do |y|
					diff = CMath.exp(m[y, x] - max) 
					(diff / sum) * (1.0 - (diff / sum))
				end

			end)
			return m.transpose
		end
	end


	# class HardMax
	# 	def self.func(m)
	# 		Matrix.set((0...m.size_y).map do |y|
	# 			ind_max = m[y].each_with_index.max[1]
	# 			out = (0...m.size_x).map do |x|
	# 				if x == ind_max
	# 					1.0
	# 				else
	# 					0.0
	# 				end
	# 			end
	# 		end)
	# 	end

	# 	def self.derivate(m)
	# 		Matrix.set((0...m.size_y).map do |y|
	# 			(0...m.size_x).map do |x|
	# 				m[y, x] * (1.0 - m[y, x])
	# 			end
	# 		end)
	# 	end
	# end

end
