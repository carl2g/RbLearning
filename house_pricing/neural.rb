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

nn.addLayer(NetLayer.new(128, data_x.size_x, ActivFunc::ReLu, lrn: 0.0000001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9)))
nn.addLayer(NetLayer.new(32, 128, ActivFunc::ReLu, lrn: 0.0000001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9)))
nn.addLayer(NetLayer.new(1, 32, ActivFunc::ReLu, lrn: 0.0000001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9)))

(0...2500).each do |ep|
	batch_x, batch_y = train.batch(y: data_y, x: data_x, batch_size: 32)
	batch_x, batch_y = batch_x.transpose, batch_y.transpose
	it = 64000
	(0..it).each do |i|
		layers = nn.train(batch_x, batch_y)
		puts "epoch: #{ep} iteration: #{i}"
		# zs, act = nn.feedForward(batch_x)
		# puts "Predictions: "
		# act.last.transpose.printM(4)
		# puts "Expected results: "
		# batch_y.transpose.printM
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
