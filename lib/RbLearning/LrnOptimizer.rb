module LrnOptimizer

	class Momentum
		attr_accessor :dwOpt, :dbOpt, :beta

		def initialize(beta: 0.0)
			@beta = beta
		end

		def setSize(size_y, size_x)
			@dwOpt = Matrix.new(size_y, size_x)
		end

		def optimize(dw, dz)
			@dwOpt = @dwOpt.applyOp(:*, @beta) + dw.applyOp(:*, 1.0 - @beta)
			@dbOpt = dz

			return [@dwOpt, @dbOpt]
		end
	end

end