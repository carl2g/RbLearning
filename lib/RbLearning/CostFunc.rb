module CostFunc

	module MeanSqrtErr
		
		def self.loss(pred, res)
			err = (0...pred.size_y).sum do |y|
				(0...pred.size_x).sum do |x|
					(pred[y, x] - res[y, x])**2
				end
			end
			return (err / (1.0 / 2.0 * pred.size_y))**0.5
		end

		# def self.log_loss(pred, res)
		# 	err = (0...pred.size_y).sum do |y|
		# 		(0...pred.size_x).sum do |x|
		# 			if pred[y, x] > 0 && res[y, x] > 0
		# 				(CMath.log(pred[y, x]) - CMath.log(res[y, x]))**2 
		# 			else
		# 				0
		# 			end
		# 		end
		# 	end
		# 	return (err / pred.size_y)**0.5
		# end


		# def self.std_loss(pred, res)
		# 	min = res.matrix.min
		# 	max = res.matrix.max
		# 	err = (0...pred.size_y).sum do |y|
		# 		(0...pred.size_x).sum do |x|
		# 			(pred[y, x] - res[y, x])**2
		# 		end
		# 	end
		# 	return (((1.0 / 2.0) * err) / pred.size_y)**0.5 / (max - min)
		# end

		def self.func(pred, res)
			m = Matrix.set((0...pred.size_y).map do |y|
				(0...pred.size_x).map do |x|
					(pred[y, x] - res[y, x])**2
				end
			end)
			return m.applyOp(:*, (1.0 / 2.0))
		end

		def self.deriv(pred, res)
			m = Matrix.set((0...pred.size_y).map do |y|
				(0...pred.size_x).map do |x|
					(pred[y, x] - res[y, x])
				end
			end)
			return m
		end


	end

	module MeanAbsErr

		def self.loss(pred, res)
			err = (0...pred.size_y).sum do |y|
				(0...pred.size_x).sum do |x|
					(pred[y, x] - res[y, x]).abs
				end
			end
			return err / pred.size_y
		end

	end

	module CrossEntropy

		def self.func(pred, res)
			Matrix.set((0...pred.size_y).map do |y|
				(0...pred.size_x).map do |x|
					if res[x, y] == 1.0
						-Math.log(pred[y, x])
					else
						-Math.log(1.0 - pred[y, x])
					end
				end
			end)
		end

		def self.deriv(pred, res)
			m = Matrix.set((0...pred.size_y).map do |y|
				(0...pred.size_x).map do |x|
					if res[y, x] == 1.0
						-1.0 / (pred[y, x])
					else
						1.0 / (1.0 - pred[y, x])
					end
				end
			end)
			return m
		end

		def self.loss(pred, res)
			err = (0...pred.size_y).sum do |y|
				(0...pred.size_x).sum do |x|
					if res[y, x] == 1.0
						-Math.log(pred[y, x])
					else
						-Math.log(1.0 - pred[y, x])
					end
				end
			end
			return err / pred.size_y
		end

	end

end
