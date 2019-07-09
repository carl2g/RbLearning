module LrnOptimizer

	class Momentum
		attr_accessor :dw, :db, :beta

		def initialize(beta: 0.9)
			@beta = beta
		end

		def setSize(size_y, size_x)
			@dwOpt = Matrix.new(size_y, size_x)
			@dbOpt = Matrix.new(1, size_x)
		end

		def optimize(dw, dz)
			@dwOpt = @dwOpt.applyOp(:*, @beta) + dw.applyOp(:*, 1 - @beta)
			@dbOpt = dz
			# @dbOpt = @dbOpt.applyOp(:*, @beta) + dz.applyOp(:*, 1 - @beta)
			[@dwOpt, @dbOpt]
		end
	end

end