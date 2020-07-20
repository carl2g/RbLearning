class NetLayer

	attr_accessor :w, :b, :activFunc, :lrn, :dropOut, :lrnOptimizer, :regularizer, :weigthInitFunc, :size

	def initialize(
			size: 1,
			activFunction: nil, 
			lrn: 0.03, 
			dropOut: 0.0,
			weigthInitFunc: lambda { return Random.rand(0..0.01) },
			lrnOptimizer: LrnOptimizer::Momentum.new, 
			regularizer: Regularizer::L2.new,
			weigths: nil
		)

		@size = size
		@weigthInitFunc = weigthInitFunc
		@activFunc = activFunction
		@lrn = lrn
		@dropOut = dropOut
		
		@lrnOptimizer = lrnOptimizer
		@regularizer = regularizer
	end

	def setWeigths(w, b)
		self.lrnOptimizer.setSize(w.size_y, w.size_x)
		self.w = w
		self.b = b
	end

	def initWeigths(rowSize, colSize)
		self.lrnOptimizer.setSize(rowSize, colSize)
		self.w = self.initWeightsFunc(rowSize, colSize, self.weigthInitFunc)
		self.b = self.initWeightsFunc(1, rowSize, self.weigthInitFunc)
	end

	def optimize(dw)
		if @lrnOptimizer.beta == 0.0
			return dw
		else
			return @lrnOptimizer.optimize(dw)
		end
	end

	def regularizeForward(w)
		return @regularizer.regularizeForward(w)
	end

	def regularizeBackward(dw, w)
		return @regularizer.regularizeBackward(dw, w)
	end

	def initWeightsFunc(size_y, size_x = size_y, func = weigthInitFunc)
		w = Matrix.new(size_y, size_x)
		(0...size_y).each do |y|
			(0...size_x).each do |x|
				w[y, x] = func.call()
			end
		end
		return w
	end
end
