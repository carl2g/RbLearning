class NetLayer

	attr_accessor :w, :b, :activFunc, :lrn, :dropOut, :lrnOptimizer

	def initialize(size_y, size_x, actFunc, lrn: 0.03, dropOut: 0.0, min: -0.1, max: 0.1, lrnOptimizer: nil)
		@w = initWeights(size_y, size_x, min, max)
		@b = initWeights(size_y, 1, 0.0, 0.0)
		@activFunc = actFunc
		@lrn = lrn
		@dropOut = dropOut
		if lrnOptimizer
			@lrnOptimizer = lrnOptimizer
			@lrnOptimizer.setSize(size_y, size_x)
		end
	end

	def optimize(dw, dz)
		if @lrnOptimizer
			@lrnOptimizer.optimize(dw, dz)
		else
			[dw, dz]
		end
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
