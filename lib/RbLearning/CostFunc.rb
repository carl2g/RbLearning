module CostFunc

	def costFunc(prob, res)
		prob - res
	end

	def crossEntropyCost(pred, res)
		Matrix.set((0...pred.size_y).map do |y|
			(0...pred.size_x).map do |x|
				if res[y, x] == 1.0
					Math.log(pred[y, x])
				else
					-Math.log(1.0 - pred[y, x])
				end
			end
		end)
	end

end