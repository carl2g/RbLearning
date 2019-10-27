require './../lib/RbLearning'
require 'benchmark'

include Statistic
include Utils
include ActivFunc

train = DataManager.new("./input/train.csv")
data_y = train.remove('SalePrice')
train.remove('Id')
train.addDumies
train.removeKeysNullVall(0.75, 0.0)

tests = DataManager.new("./input/test.csv")
tests.remove('Id')
tests.addDumies

nn = NeuroNet.new

remove_lab = []

data = train.matrix
tr_data = data.transpose

(0...tr_data.size_y).each do |i|
	puts "#{train.labels[i]} #{Statistic::corelation(tr_data[i], data_y).round(3)}"

	if Statistic::corelation(tr_data[i], data_y).round(3).abs < 0.15
		remove_lab << train.labels[i]
	end
end

remove_lab.each do |lab|
	train.remove(lab)
end

puts "=" * 20

data = train.matrix
tr_data = data.transpose

(0...tr_data.size_y).each do |i|
	puts "#{train.labels[i]} #{Statistic::corelation(tr_data[i], data_y).round(3)}"
end

train.keepLabels(tests.labels)
tests.keepLabels(train.labels)

data_x = Statistic::normalize_range(train.matrix, axis: 1)
data_y = Matrix.setVectorizedMatrix(data_y, data_y.size, 1)


layers = [
	NetLayer.new(
	 	128, 
	 	data_x.size_x, 
	 	ActivFunc::ReLu, 
	 	lrn: 0.000000001, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
	),
	NetLayer.new(
	 	32, 
	 	128, 
	 	ActivFunc::ReLu, 
	 	lrn: 0.000000001, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
	),
	NetLayer.new(
	 	1, 
	 	32, 
	 	ActivFunc::ReLu, 
	 	lrn: 0.000000001, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
	)
]

nn.addLossFunc(LossFunc::MeanSqrtErr)
nn.addLayers(layers)
nn.train([data_y, data_x], batch_size: 32, iteration: 24, epoch: 150)


data_x = Statistic::normalize_range(tests.matrix, axis: 1)
zs, pred = nn.feedForward(data_x)

CSV.open("./res.csv", "wb") do |csv|
	csv << ['Id', 'SalePrice']
	m = pred.last
	(0...m.size_y).each do |i|
		csv <<  [1461 + i, m[i][0]]
	end
end
