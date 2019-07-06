module LossFunc

	module MeanSqrtErr
		def self.func(pred, res)
			err = (0...pred.size_y).sum do |y|
				(0...pred.size_x).sum do |x|
					(pred[y, x] - res[y, x])**2
				end
			end
			return (err / pred.size_y)**0.5
		end
	end

	module MeanAbsErr
		def self.func(pred, res)
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
