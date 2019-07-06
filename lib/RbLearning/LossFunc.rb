module LossFunc

	def costFunc(prob, res)
		prob - res
	end

	def meanSqrtErr(pred, res)
		err = 0
		(0...pred.size_y).each do |y|
			(0...pred.size_x).each do |x|
				err += (pred[y, x] - res[y, x])**2
			end
		end
		return (err / pred.size_y)**0.5
	end

	def meanAbsErr(pred, res)
		err = 0
		(0...pred.size_y).mpa do |y|
			(0...pred.size_x).map do |x|
				err += (pred[y, x] - res[y, x]).abs
			end
		end
		return err / pred.size_y
	end

	module CrossEntropy

		def self.func(pred, res)
			Matrix.set((0...pred.size_y).map do |y|
				[(0...pred.size_x).sum do |x|
					if res[y, x] == 1.0
						-Math.log(pred[y, x])
					else
						-Math.log(1.0 - pred[y, x])
					end
				end]
			end)
		end

		def self.derivate(pred, res)
			Matrix.set((0...pred.size_y).map do |y|
				(0...pred.size_x).map do |x|
					if res[y, x] == 1.0
						-1.0 / (pred[y, x])
					else
						-1.0 / (1.0 - pred[y, x])
					end
				end
			end)
		end

	end

end
