class NetLayer

	attr_accessor :w, :b, :activFunc, :lrn, :dropOut

	def initialize(size_y, size_x, actFunc, lrn, dropOut = 0, min = 1.0, max = 1.0)
		@w = initWeights(size_y, size_x, min, max)
		@b = initWeights(1, size_x, min, max)
		@activFunc = actFunc
		@lrn = lrn
		@dropOut = dropOut
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
