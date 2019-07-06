module ActivFunc

	module ReLu
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

	module Sigmoid
		def self.sigmoidUnit(x)
			1.0 / (1.0 + CMath.exp(-x))
		end

		def self.func(m)
			m.printM
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
			      	CMath.exp(-x) / (1.0 + CMath.exp(-x))**2
				end,
				m.size_y,
				m.size_x
			)
		end
	end

	module Tanh
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

	module SoftMax
		def self.func(m)
			Matrix.set((0...m.size_y).map do |y|
				sum = 0.0
				max = m[y].max
				out = (0...m.size_x).map do |x|
					tmp = CMath.exp(m[y, x] - max)
					sum += tmp
					tmp
				end
				out.map { |val| (val / sum) }
			end)
		end

		def self.derivate(m)
			Matrix.set((0...m.size_y).map do |y|
				max = m[y].max
				sum = (0...m.size_x).sum do |i|
					CMath.exp(m[y, i] - max)
				end
				(0...m.size_x).map do |x|
					(CMath.exp(m[y, x] - max) / sum) * (1.0 - (CMath.exp(m[y, x] - max) / sum))
				end
			end)
		end
	end


	# module HardMax
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
