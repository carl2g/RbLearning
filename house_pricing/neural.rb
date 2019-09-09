require './../lib/RbLearning'
require 'benchmark'

include Statistics
include Utils
include ActivFunc

train = DataManager.new("./input/train.csv")
data_y = train.remove('SalePrice')
train.removeKeysNullVall(0.2)
train.addDumies

tests = DataManager.new("./input/test.csv")

tests.removeKeysNullVall(0.2)
tests.addDumies

train.keepLabels(tests.labels)
tests.keepLabels(train.labels)

nn = NeuroNet.new

data_x = train.matrix.normalize(axis: 1)
data_y = Matrix.setVectorizedMatrix(data_y, data_y.size, 1)

layers = [
	NetLayer.new(
	 	10, 
	 	data_x.size_x, 
	 	ActivFunc::ReLu, 
	 	lrn: 0.0001, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
		# regularizer: Regularizer::DropOut.new(dropOutRate: 0.25),
		regularizer: Regularizer::L2.new(alpha: 1000000000000)
	),
	# NetLayer.new(
	#  	1, 
	#  	data_x.size_x, 
	#  	ActivFunc::ReLu, 
	#  	lrn: 0.0001, 
	# 	lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
	# 	# regularizer: Regularizer::DropOut.new(dropOutRate: 0.25),
		# regularizer: Regularizer::L1.new(alpha: 4000)
	# ),
	NetLayer.new(
	 	1, 
	 	10, 
	 	ActivFunc::ReLu, 
	 	lrn: 0.0001, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
		# regularizer: Regularizer::DropOut.new(dropOutRate: 0.25),
		regularizer: Regularizer::L2.new(alpha: 1)
	)
]

nn.addLossFunc(LossFunc::MeanSqrtErr)
nn.addLayers(layers)

(0...25000).each do |ep|
	batch_x, batch_y = train.batch(y: data_y, x: data_x, batch_size: 24)
	batch_x, batch_y = batch_x.transpose, batch_y.transpose
	it = 32000000
	(0..it).each do |i|
		layers = nn.train(batch_x, batch_y)
		batch_x.printShape
		puts "epoch: #{ep} iteration: #{i}"
		# zs, act = nn.feedForward(batch_x)
		# puts "Predictions: "
		# act.last.transpose.printM(4)
		# puts "Expected results: "
		# batch_y.transpose.printM
		# nn.layers.last.w.printM
		# puts nn.layers.last.w.sum
		puts "=" * 30
	end
end

it = 1000
(0..it).each do |i|
	batch_x, batch_y = data_x.transpose, data_y.transpose
	nn.train(batch_x, batch_y)
end

data_x = tests.matrix

zs, pred = nn.feedForward(data_x.transpose)

CSV.open("./res.csv", "wb") do |csv|
	csv << ['Id', 'SalePrice']
	m = pred.last.transpose
	(0...m.size_y).each do |i|
		csv <<  [1461 + i, m[i][0]]
	end
end
