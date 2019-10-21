require './../lib/RbLearning'
require 'benchmark'

include Statistic
include Utils
include ActivFunc

train = DataManager.new("./inputs/train.csv")
train.remove('id')
data_y = train.remove('target')

nn = NeuroNet.new(
	# lossFunction: LossFunc::CrossEntropy
)

data_x = train.matrix.normalize(axis: 1)
data_y = Matrix.set(data_y, size_x: 1, size_y: data_y.size)

layers = [
	NetLayer.new(
		32, 
		data_x.size_x, 
		ActivFunc::LeakyReLu, 
		lrn: 0.01, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
		# regularizer: Regularizer::DropOut.new(dropOutRate: 0.35)
	),
	NetLayer.new(
		1, 
		32, 
		ActivFunc::Sigmoid,
		lrn: 0.01, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99)
	)
]

nn.addLayers(layers)
nn.train([data_y, data_x], batch_size: 32, iteration: 420000, epoch: 500)


nn.layers.each do |l|
	l.w.printM
end

tests = DataManager.new("./inputs/test.csv")
ids = tests.remove('id')
m = tests.matrix.normalize(axis: 1)

pred = nn.predict(m.transpose)

CSV.open("./res2.csv", "wb") do |csv|
	csv << ['id', 'target']
	m = pred.transpose
	(0...m.size_y).each do |i|
		csv <<  [ids[i], m[i][0].round]
	end
end
