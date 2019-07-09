require './../lib/RbLearning'
require 'benchmark'

include Statistics
include Utils
include ActivFunc

train = DataManager.new("./input/lol.csv")
data_y = train.remove('label')

nn = NeuroNet.new

data_x = train.matrix
data_y = Matrix.set(data_y.map do |i|
	tmp = [0] * 10
	tmp[i.to_i] = 1
	tmp
end)

nn.addLayer(NetLayer.new(data_x.size_x, 64, ActivFunc::ReLu, lrn: 0.0001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9)))
nn.addLayer(NetLayer.new(64, 10, ActivFunc::SoftMax, lrn: 0.0001, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.9)))
nn.addLossFunc(LossFunc::CrossEntropy)

(0...300).each do |ep|
	batch_x, batch_y = train.batch(data_y, data_x: data_x, batch_size: 3)
	it = 24000
	# it = 64
	(0..it).each do |i|
		# batch_x, batch_y = data_x, data_y
		layers = nn.train(batch_x, batch_y)
		puts "epoch: #{ep} iteration: #{i}"
		# break if nn.lastLoss < 1.0
		zs, act = nn.feedForward(batch_x)
		act.last.printM(4)
		batch_y.printM
		puts "=" * 30
	end
end

(0...400).each do |ep|
	batch_x, batch_y = train.batch(data_y, data_x: data_x, batch_size: 32)
	# it = 24
	it = 1
	(0..it).each do |i|
		batch_x, batch_y = data_x, data_y
		layers = nn.train(batch_x, batch_y)
		puts "epoch: #{ep} iteration: #{i}"
		# break if nn.lastLoss < 1.0
		# zs, act = nn.feedForward(batch_x)
		# act.last.printM(4)
		# batch_y.printM
		# puts "=" * 30
	end
end

tests = DataManager.new("./input/test.csv")
m = tests.matrix

x = Matrix.setVectorizedMatrix(m[0...m.size_y * m.size_x], m.size_y, m.size_x)

zs, pred = nn.feedForward(x)

CSV.open("./res3.csv", "wb") do |csv|
	csv << ['ImageId', 'Label']
	m = pred.last.get2DArr
	(0...pred.last.size_y).each do |i|
		csv <<  [i + 1, m[i].each_with_index.max[1]]
	end
end
