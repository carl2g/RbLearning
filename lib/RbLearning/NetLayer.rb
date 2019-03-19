class NetLayer

	attr_accessor :w, :b, :actFunc, :primeActFunc, :lrn, :dropOut

	def initialize(size_y, size_x, actFunc, lrn, dropOut = 0, min = 0.01, max = 0.01)
		@w = initWeights(size_y, size_x, min, max)
		@b = initWeights(size_x, 1, min, max)
		@actFunc = actFunc
		@primeActFunc = actFunc + "Prime"
		@lrn = lrn
		@dropOut = dropOut
	end

	def initWeights(size_y, size_x = size_y, min = -1.0, max = 1.0)
		r = Random.new
		w = Matrix.new(size_y, size_x)
		(0...size_y).each do |y|
			(0...size_x).each do |x|
				w[y, x] = r.rand(min..max)
			end
		end
		return w
	end
end
