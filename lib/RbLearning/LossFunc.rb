module LossFunc

	module MeanSqrtErr
		
		def self.loss(pred, res)
			err = (0...pred.size_y).sum do |y|
				(0...pred.size_x).sum do |x|
					(pred[y, x] - res[y, x])**2
				end
			end
			return (err / pred.size_y)**0.5
		end

		def self.func(pred, res)
			Matrix.set((0...pred.size_y).map do |y|
				(0...pred.size_x).map do |x|
					(0...pred.size_x).map do |x|
						(pred[y, x] - res[y, x])**2
					end
				end
			end)
		end

		def self.deriv(pred, res)
			Matrix.set((0...pred.size_y).map do |y|
				(0...pred.size_x).map do |x|
					(0...pred.size_x).map do |x|
						2 * (pred[y, x] - res[y, x])
					end
				end
			end)
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
			Matrix.set((0...pred.size_x).map do |x|
				(0...pred.size_y).map do |y|
					if res[x, y] == 1.0
						-Math.log(pred[y, x])
					else
						-Math.log(1.0 - pred[y, x])
					end
				end
			end).transpose
		end

		def self.deriv(pred, res)
			m = Matrix.set((0...pred.size_x).map do |x|
				(0...pred.size_y).map do |y|
					if res[y, x] == 1.0
						-1.0 / pred[y, x]
					else
						1.0 / (1.0 - pred[y, x])
					end
				end
			end)
			return m.transpose
		end

		def self.loss(pred, res)
			err = (0...pred.size_x).sum do |x|
				(0...pred.size_y).sum do |y|
					if res[y, x] == 1.0
						-Math.log(pred[y, x])
					else
						-Math.log(1.0 - pred[y, x])
					end
				end
			end
			return err / pred.size_x
		end

	end

end
