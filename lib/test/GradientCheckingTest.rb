require 'test/unit'
require './lib/RbLearning'

class GradientCheckingTest < Test::Unit::TestCase
	EPSYLON = 0.00001

	def test_1
		train_x = Matrix.set([[4, 6, 7], [6, 8, 10]])
		train_y = Matrix.set([[10], [12]])
		lrn = 0.01

		layers = [
			Input.new(train_x.size_x),
			NetLayer.new(
				size: 1,
			 	lrn: lrn,
	 		activFunction: ActivFunc::ReLu
			)
		]
		net = NeuroNet.new(costFunction: CostFunc::MeanSqrtErr)
		net.addLayers(layers)

		original_w = layers.last.w.copy

		layers.last.b = layers.last.b.applyOp(:*, 0)

		(0...layers.last.w.size_x).each do |i|
			
			layers.last.w = original_w.copy
			layers.last.w[0, i] -= EPSYLON
			zs, act = net.feedForward(train_x)
			a = net.costFunc.func(act.last, train_y)

			layers.last.w = original_w.copy
			layers.last.w[0, i] += EPSYLON
			zs, act = net.feedForward(train_x)
			b = net.costFunc.func(act.last, train_y)

			delt = (b - a).applyOp(:*, lrn / (2 * EPSYLON)).sumOrd
			updated_w = (original_w - delt).applyOp(:round, 6)
			
			puts "==========delta========"
			delt.printM(5)

			layers.last.w = original_w.copy
			layers, loss = net.internal_train(train_y, train_x)
			layers.last.w = layers.last.w.applyOp(:round, 6)

			puts "========== W after training ========"
			puts layers.last.w[0, i]

			puts "========== Calculated new W ========"
			puts updated_w[0, i]
			puts "=" * 80
			res = updated_w[0, i] == layers.last.w[0, i]
	    	assert(res, "Derivation is not the same")
	    end
	end

end