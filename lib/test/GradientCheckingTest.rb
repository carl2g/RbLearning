require 'test/unit'
require './lib/RbLearning'

class GradientCheckingTest < Test::Unit::TestCase
	EPSYLON = 0.000000001

	def test_1
		train_x = Matrix.set([
			[4, 6, 7], 
			[6, 8, 10],
			[2, 32, 14]
		])
		train_y = Matrix.set([
			[10], 
			[12],
			[32]
		])
		lrn = 0.01

		layers = [
			Input.new(train_x.size_x),
			NetLayer.new(
				size: 64,
			 	lrn: lrn,
	 			activFunction: ActivFunc::ReLu
			),
			NetLayer.new(
				size: 1,
				lrn: lrn,
		 		activFunction: ActivFunc::Sigmoid
			)
		]
		net = NeuroNet.new(costFunction: CostFunc::MeanSqrtErr)
		net.addLayers(layers)


		layers.each do |l|
			original_w = l.w.copy
			(0...l.w.size_y).each do |j|
				(0...l.w.size_x).each do |i|
					
					l.w = original_w.copy
					l.w[j, i] -= EPSYLON
					zs, act = net.feedForward(train_x)
					a = net.costFunc.func(act.last, train_y)

					l.w = original_w.copy
					l.w[j, i] += EPSYLON
					zs, act = net.feedForward(train_x)
					b = net.costFunc.func(act.last, train_y)

					delt = (b - a).applyOp(:*, lrn / (2.0 * EPSYLON)).sumOrd
					updated_w = (original_w - delt).applyOp(:round, 3)
					
					puts "==========delta w(#{j})(#{i}) ========"
					delt.printM(5)

					l.w = original_w.copy
					layers, loss = net.internal_train(train_y, train_x)
					l.w = l.w.applyOp(:round, 3)

					puts "========== W(#{j})(#{i}) after training ========"
					puts l.w[j, i]

					puts "========== Manualy calculated updated W(#{j})(#{i}) ========"
					puts updated_w[j, i]
					puts "=" * 80
					res = updated_w[j, i] <= l.w[j, i] + 0.01 && updated_w[j, i] >= l.w[j, i] - 0.01
			    	assert(res, "Derivation is not the same")
			    end
			end
		end
	end

end