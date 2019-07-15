require './../lib/RbLearning'
require 'benchmark'

include Statistics
include Utils
include ActivFunc

train = DataManager.new("./input/lol.csv")
data_y = train.remove('label')

nn = NeuroNet.new

data_x = train.matrix.normalize
data_y = Matrix.set(data_y.map do |i|
	tmp = [0] * 10
	tmp[i.to_i] = 1
	tmp
end)

# nn.addLayer(NetLayer.new(64, data_x.size_x, ActivFunc::ReLu, lrn: 0.1, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.95)))
nn.addLayer(NetLayer.new(10, data_x.size_x, ActivFunc::SoftMax, lrn: 0.1, lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99)))
nn.addLossFunc(LossFunc::CrossEntropy)

# (0...300).each do |ep|
# 	batch_x, batch_y = train.batch(y: data_y, x: data_x, batch_size: 24)
# 	batch_x, batch_y = batch_x.transpose, batch_y.transpose
# 	it = 62
# 	# it = 64
# 	(0..it).each do |i|
# 		# batch_x, batch_y = data_x, data_y
# 		layers = nn.train(batch_x, batch_y)
# 		puts "epoch: #{ep} iteration: #{i}"
# 		# zs, act = nn.feedForward(batch_x)
# 		# act.last.transpose.printM(4)
# 		# batch_y.transpose.printM
# 		puts "=" * 30
# 	end
# end

(0...1).each do |ep|
	batch_x, batch_y = data_x.transpose, data_y.transpose

	# it = 24
	it = 1
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

tests = DataManager.new("./input/test.csv")
m = tests.matrix

x = Matrix.setVectorizedMatrix(m[0...m.size_y * m.size_x], m.size_y, m.size_x)

zs, pred = nn.feedForward(x.transpose)

CSV.open("./res2.csv", "wb") do |csv|
	csv << ['ImageId', 'Label']
	m = pred.last
	(0...pred.last.transpose.size_y).each do |i|
		csv <<  [i + 1, m[i].each_with_index.max[1]]
	end
end
