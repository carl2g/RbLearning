module Regularizer
	
	class DropOut

		attr_accessor :dropOutRate, :filter
		
		def initialize(dropOutRate: 0.0)
			@dropOutRate = dropOutRate
		end

		def regularizeForward(w)
			return w if @dropOutRate == 0.0

			@filter = Matrix.new(w.size_y, w.size_x, 1)
			Random.srand
			size = (@dropOutRate.to_f * w.size_y * w.size_x).round
			(0...size).each do |drop_count|
				y = Random.rand(0...w.size_y)
				x = Random.rand(0...w.size_x)
				@filter[y, x] = 0
			end
			return w ** @filter
		end

		def regularizeBackward(dw, w)
			return dw if @dropOutRate == 0.0

			return dw ** @filter
		end

	end	

	class L1
		attr_accessor :alpha


		def initialize(alpha: 0.0)
			@alpha = alpha
		end

		def regularizeForward(w)
			return w if @alpha == 0.0

			return w
		end

		def regularizeBackward(dw, w)
			return dw if @alpha == 0.0

			return dw + w.applyOp(:abs, 2).applyOp(:*, alpha)
		end

	end

	class L2
		attr_accessor :alpha

		def initialize(alpha: 0.0)
			@alpha = alpha
		end

		def regularizeForward(w)
			return w if @alpha == 0.0

			return w.applyOp(:**, 2).applyOp(:*, alpha)
		end

		def regularizeBackward(dw, w)
			return dw if @alpha == 0.0

			return dw + w.applyOp(:*, alpha * 2)
		end



	end

end