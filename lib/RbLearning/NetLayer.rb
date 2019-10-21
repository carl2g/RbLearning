class NetLayer

	attr_accessor :w, :b, :activFunc, :lrn, :dropOut, :lrnOptimizer, :regularizer

	def initialize(
			size_y, 
			size_x, 
			actFunc, 
			lrn: 0.03, 
			dropOut: 0.0, 
			min: -0.001, 
			max: 0.001, 
			lrnOptimizer: LrnOptimizer::Momentum.new, 
			regularizer: Regularizer::L2.new
		)

		@w = initWeights(size_y, size_x, min, max)
		@b = Matrix.new(1, size_y, 1)
		
		@activFunc = actFunc
		@lrn = lrn
		@dropOut = dropOut
		
		@lrnOptimizer = lrnOptimizer
		@lrnOptimizer.setSize(size_y, size_x)

		@regularizer = regularizer
	end

	def optimize(dw, dz)
		if @lrnOptimizer.beta == 0.0
			return [dw, dz]
		else
			return @lrnOptimizer.optimize(dw, dz)
		end
	end

	def regularizeForward(w)
		return @regularizer.regularizeForward(w)
	end

	def regularizeBackward(dw, w)
		return @regularizer.regularizeBackward(dw, w)
	end

	def reset
		@w =  initWeights(@w.size_y, @w.size_x, min, max)
		@b =  initWeights(@b.size_y, @b.size_x, min, max)
		return [@w, @b]
	end

	def initWeights(size_y, size_x = size_y, min = -1.0, max = 1.0)
		w = Matrix.new(size_y, size_x)
		(0...size_y).each do |y|
			Random.srand
			(0...size_x).each do |x|
				w[y, x] = Random.rand(min..max)
			end
		end
		return w
	end
end
