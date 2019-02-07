class NetLayer

	attr_accessor :w, :b, :actFunc, :primeActFunc

	def initialize(size_y, size_x, actFunc = 'sigmoid', min = -2.7, max = 2.7, primeActFunc = actFunc + "Prime")
		@w = initWeights(size_y, size_x, min, max)
		@b = initWeights(size_x, 1, min, max)
		@actFunc = actFunc
		@primeActFunc = primeActFunc
	end

	def initWeights(size_y, size_x = size_y, min = -1.0, max = 1.0)
		r = Random.new
		w = Matrix.new(size_y, size_x)
		(0...size_y).each do |y|
			(0...size_x).each do |x|
				w[y, x] = r.rand(min...max)
			end
		end
		return w
	end
end