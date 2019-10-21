require './../lib/RbLearning'
require 'benchmark'

include Statistic
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

data_x = Statistic::normalize_range(train.matrix, axis: 1)
data_y = Matrix.setVectorizedMatrix(data_y, data_y.size, 1)

layers = [
	NetLayer.new(
	 	64, 
	 	data_x.size_x, 
	 	ActivFunc::ReLu, 
	 	lrn: 0.000000001, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9),
	),
	NetLayer.new(
	 	1, 
	 	64, 
	 	ActivFunc::ReLu, 
	 	lrn: 0.000000001, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9),
	)
]

nn.addLossFunc(LossFunc::MeanSqrtErr)
nn.addLayers(layers)
nn.train([data_y, data_x], batch_size: 32, iteration: 42, epoch: 100)


data_x = Statistic::normalize_range(tests.matrix, axis: 1)
zs, pred = nn.feedForward(data_x)

CSV.open("./res.csv", "wb") do |csv|
	csv << ['Id', 'SalePrice']
	m = pred.last.transpose
	(0...m.size_y).each do |i|
		csv <<  [1461 + i, m[i][0]]
	end
end
