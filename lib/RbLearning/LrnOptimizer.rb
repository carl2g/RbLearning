module LrnOptimizer

	class Momentum
		attr_accessor :dwOpt, :beta

		def initialize(beta: 0.0)
			@beta = beta
		end

		def setSize(size_y, size_x)
			@dwOpt = Matrix.new(size_y, size_x)
		end

		def optimize(dw)
			@dwOpt = @dwOpt.applyOp(:*, @beta) + dw.applyOp(:*, 1.0 - @beta)
			return @dwOpt
		end
	end

end