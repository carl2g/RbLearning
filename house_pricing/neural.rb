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

nn.addLayer(NetLayer.new(data_x.size_x, 64, ActivFunc::ReLu, lrn: 0.00001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.25)))
nn.addLayer(NetLayer.new(64, 12, ActivFunc::ReLu, lrn: 0.00001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.0)))
nn.addLayer(NetLayer.new(12, 1, ActivFunc::ReLu, lrn: 0.00001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.0)))

# (0...500).each do |ep|
# 	batch_x, batch_y = train.batch(data_y, data_x: data_x, batch_size: 32)
# 	it = 64
# 	(0..it).each do |i|
# 		layers = nn.train(batch_x, batch_y)
# 		puts "epoch: #{ep} iteration: #{i}"
# 		# zs, act = nn.feedForward(batch_x)
# 		# puts "Predictions: "
# 		# act.last.printM(4)
# 		# puts "Expected results: "
# 		# batch_y.printM
# 		# puts "=" * 30
# 	end
# end

	it = 2500
	(0..it).each do |i|
		layers = nn.train(data_x, data_y)
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
