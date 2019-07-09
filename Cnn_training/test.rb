require './../lib/RbLearning'
require 'benchmark'

include Statistics
include Utils
include ActivFunc
include LossFunc::CrossEntropy


data_x = Matrix.set([
	[1, 1, 1],
	[4, 3, 2],
	[5, 3, 2]
])

data_y = Matrix.set([
	[1],
	[4],
	[5]
])

nn = NeuroNet.new
nn.addLayer(NetLayer.new(data_x.size_x, 12, ActivFunc::ReLu, lrn: 0.0001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99)))
nn.addLayer(NetLayer.new(12, 1, ActivFunc::ReLu, lrn: 0.0001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99)))

(0...5000).each do |ep|
	# batch_x, batch_y = train.batch(data_y, data_x: data_x, batch_size: 32)
	# it = 24
	it = 240000
	(0..it).each do |i|
		batch_x, batch_y = data_x, data_y
		layers = nn.train(batch_x, batch_y)
		puts "epoch: #{ep} iteration: #{i}"
		# break if nn.lastLoss < 1.0
		zs, act = nn.feedForward(batch_x)
		act.last.printM(4)
		batch_y.printM
		puts "=" * 30
	end
end