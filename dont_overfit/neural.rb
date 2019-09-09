require './../lib/RbLearning'
require 'benchmark'

include Statistics
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
	# NetLayer.new(
	# 	64, 
	# 	data_x.size_x, 
	# 	ActivFunc::LeakyReLu, 
	# 	lrn: 0.1, 
	# 	lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
	# 	# regularizer: Regularizer::DropOut.new(dropOutRate: 0.35)
	# 	regularizer: Regularizer::L2.new(alpha: 2)
	# ),
	# NetLayer.new(
	# 	1, 
	# 	64, 
	# 	ActivFunc::Sigmoid, 
	# 	lrn: 0.1, 
	# 	lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
	# 	regularizer: Regularizer::L2.new(alpha: 2)
	# )
	NetLayer.new(
		1, 
		data_x.size_x, 
		ActivFunc::Sigmoid,
		lrn: 0.02, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
		regularizer: Regularizer::L1.new(alpha: 100)
	)
]

nn.addLayers(layers)

(0...1000).each do |ep|
	batch_x, batch_y = train.batch(y: data_y, x: data_x, batch_size: 32)
	batch_x, batch_y = batch_x.transpose, batch_y.transpose
	it = 420000
	(0..it).each do |i|
		# batch_x, batch_y = data_x, data_y
		layers = nn.train(batch_x, batch_y)
		puts "epoch: #{ep} iteration: #{i}"
		# zs, act = nn.feedForward(batch_x)
		# act.last.transpose.printM(6)
		# batch_y.transpose.printM
		# layers.last.w.printM(4)
		puts "=" * 30
	end
end


(0...1).each do |ep|
	batch_x, batch_y = data_x.transpose, data_y.transpose

	# it = 24
	it = 3000
	(0..it).each do |i|
		layers = nn.train(batch_x, batch_y)
		puts "epoch: #{ep} iteration: #{i}"
		# break if nn.lastLoss < 1.0
		# zs, act = nn.feedForward(batch_x)
		# act.last.printM(4)
		# batch_y.printM
		# puts "=" * 30
	end
end


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
