module Regularizer
	
	# class DropOut

	# 	attr_accessor :dropOutRate
		
	# 	def initialize(dropOutRate: 0.0)
	# 		@dropOutRate = dropOutRate
	# 	end

	# 	def backward(dw, w)
	# 		return dw if @dropOutRate == 0.0
			
	# 		filter = Matrix.new(w.size_y, w.size_x, 1)
	# 		Random.srand
	# 		size = (@dropOutRate.to_f * w.size_y * w.size_x).round
	# 		(0...size).each do |drop_count|
	# 			y = Random.rand(0...w.size_y)
	# 			x = Random.rand(0...w.size_x)
	# 			filter[y, x] = 0
	# 		end

	# 		return dw ** filter
	# 	end

	# end	

	class L1
		attr_accessor :alpha
		
		def initialize(alpha: 0.0)
			@alpha = alpha
		end

		def forward(loss, w)
			return loss if self.alpha == 0
			return loss + w.applyOp(:**, 2).applyOp(:**, 0.5).sumAxis.applyOp(:*, self.alpha)
		end

		def backward(dw, w)
			return dw if @alpha == 0.0
			
			abs_w = Matrix.setVectorizedMatrix(
				w.matrix.map do |w|
					if w > 0
						self.alpha
					elsif w < 0
						-self.alpha
					else
						0
					end
				end, 
				w.size_y,
				w.size_x
			)
			return dw + abs_w.sumAxis
		end

	end

	class L2
		attr_accessor :alpha

		def initialize(alpha: 0.0)
			@alpha = alpha
		end

		def forward(loss, w)
			return loss if self.alpha == 0.0
			return loss + w.applyOp(:**, 2).applyOp(:*, self.alpha).sumAxis
		end

		def backward(dw, w)
			return dw if @alpha == 0.0
			return dw + w.applyOp(:*, 2 * self.alpha).sumAxis
		end



	end

end