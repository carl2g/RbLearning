module LossFunc

	def costFunc(prob, res)
		prob - res
	end

	def meanSqrErr(pred, res)
		err = 0
		(0...pred.size_y).each do |y|
			(0...pred.size_x).each do |x|
				err += (pred[y, x] - res[y, x])**2
			end
		end
		return err / pred.size_y
	end

	def meanAbsErr(pred, res)
		err = 0
		(0...pred.size_y).each do |y|
			(0...pred.size_x).each do |x|
				err += (pred[y, x] - res[y, x]).abs
			end
		end
		return err / pred.size_y
	end



end
