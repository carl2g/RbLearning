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

data_x = train.matrix
data_y = Matrix.setVectorizedMatrix(data_y, data_y.size, 1)

nn.addLayer(NetLayer.new(64, data_x.size_x, ActivFunc::ReLu, lrn: 0.001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9)))
nn.addLayer(NetLayer.new(1, 64, ActivFunc::ReLu, lrn: 0.001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9)))
# nn.addLayer(NetLayer.new(1, 32, ActivFunc::ReLu, lrn: 0.001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9)))

(0...2500).each do |ep|
	batch_x, batch_y = train.batch(y: data_y, x: data_x, batch_size: 32)
	batch_x, batch_y = batch_x.transpose, batch_y.transpose
	it = 64
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

it = 500
data_x = Matrix.set((0...800).map { |i| data_x[i] })
data_y = Matrix.set((0...800).map { |i| data_y[i] })

(0..it).each do |i|
	batch_x, batch_y = data_x.transpose, data_y.transpose
	nn.train(batch_x, batch_y)
end

data_x = tests.matrix

zs, pred = nn.feedForward(data_x)

CSV.open("./res.csv", "wb") do |csv|
	csv << ['Id', 'SalePrice']
	m = pred.last
	(0...pred.last.size_y).each do |i|
		csv <<  [1461 + i, m[i][0]]
	end
end
