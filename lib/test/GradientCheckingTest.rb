require 'test/unit'
require './lib/RbLearning'

class GradientCheckingTest < Test::Unit::TestCase
	EPSYLON = 0.0001

	def test_1
		train_x = Matrix.set([
			[0, 0], 
			[1, 1],
			[1, 0],
			[0, 1]
		])
		train_y = Matrix.set([
			[1, 0], 
			[1, 0],
			[0, 1],
			[0, 1]
		])
		
		lrn = 0.01

		layers = [
			InputLayer.new([1, train_x.size_x]),
			# NetLayer.new(
			# 	size: 4,
			#  	lrn: lrn,
	 	# 		activFunction: ActivFunc::ReLu
			# ),
			NetLayer.new(
				size: 2,
				lrn: lrn,
		 		activFunction: ActivFunc::SoftMax
			)
		]
		net = NeuroNet.new(costFunction: CostFunc::CrossEntropy)
		net.addLayers(layers)

		layers.reverse.each_with_index do |l, i|
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

					updated_w = (original_w[j, i] - delt[0, i]).round(5)
	
					puts "========== Original w(#{j})(#{i}) ========"
					puts original_w[j, i].round(5)

					puts "========== Delta w(#{j})(#{i}) ========"
					delt.printM(5)
					# train_y.printM

					l.w = original_w.copy
					net.internal_train(train_y, train_x)
					l.w = l.w.applyOp(:round, 5)

					puts "========== W(#{j})(#{i}) after training ========"
					puts l.w[j, i]

					puts "========== Manualy calculated updated W(#{j})(#{i}) ========"
					puts updated_w
					puts "=" * 80
					res = updated_w == l.w[j, i]
			    	assert(res, "Derivation is not the same")
			    end
			end
		end
	end

	# def test_2
	# 	dir = File.dirname(__FILE__)
	# 	dm = DataManager.new("#{dir}/input/train_digits.csv")
	# 	tmp = dm.remove('label')
	# 	train_y = Matrix.set(
	# 		tmp.map do |v|
	# 			(0..9).map {|i| i == v.to_i ? 1 : 0 }
	# 		end
	# 	)

	# 	train_x = dm.matrix
	# 	train_x = train_x.applyOp(:/, 255)

	# 	train_x = (0...train_x.size_y).map do |y|
	# 		[Matrix.setVectorizedMatrix(train_x[y], 28, 28)]
	# 	end

	# 	input_size = [28, 28]

	# 	lrn = 0.1

	# 	layers = [
	# 		InputLayer.new(input_size),
	# 		ConvLayer.new(filter_nb: 1, size: [1, 1], step_y: 1, step_x: 1, lrn: lrn),
	# 		# ConvLayer.new(filter_nb: 1, size: [2, 2], step_y: 2, step_x: 2, lrn: 0.1),
	# 		# MaxPoolLayer.new(size: [4, 4], step_x: 4, step_y: 4),
	# 		NetLayer.new(
	# 			size: 10,
	# 		 	activFunction: ActivFunc::SoftMax, 
	# 		 	lrn: lrn
	# 		)
	# 	]

	# 	net = Cnn.new
	# 	net.addLayers(layers)

	# 	(0...net.layers.first.filters.size).each do |i|
	# 		f = net.layers.first.filters[i]
	# 		orig_w = f.copy
	# 		puts "========== Original weigths =========="
	# 		f.printM(6)
	# 		(0...f.size_y).each do |y|
	# 			(0...f.size_x).each do |x|
					
	# 				f_in, f_out, zs, act = net.forward_step(train_x, train_y)
	# 				a = net.ann.costFunc.func(act.last, train_y)
					
	# 				net.layers.first.filters[i][y , x] = net.layers.first.filters[i][y , x] + EPSYLON
					
	# 				f_in, f_out, zs, act =  net.forward_step(train_x, train_y)
	# 				b = net.ann.costFunc.func(act.last, train_y)
					
	# 				delt = (b - a).applyOp(:*, lrn / (2.0 * EPSYLON)).sumOrd
	# 				updated_w = (orig_w[y, x] - delt[0, 0]).round(5)
	# 				puts "========== Gradient =========="
	# 				delt.printM(6)

	# 				net.layers.first.filters[i][y, x] = orig_w[y, x]

	# 				net.one_iteration_train(train_x, train_y)

						
	# 				puts "========== Manualy updated weigths =========="
	# 				puts updated_w
	# 				puts "========== Gradient descent updated weigths =========="
	# 				net.layers.first.filters[i].printM(5)

	# 				res = updated_w == net.layers.first.filters[i][y, x]
	# 		    	assert(res, "Derivation is not the same")
	# 			end
	# 		end
	# 	end
	# end

end