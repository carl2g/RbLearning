require './../lib/RbLearning'

dm = DataManager.new("./input/train_digits.csv")
tmp = dm.remove('label')
train_y = Matrix.set(
	tmp.map do |v|
		(0..9).map {|i| i == v.to_i ? 1 : 0 }
	end
)

train_x = dm.matrix
train_x = train_x.applyOp(:/, 255)

train_x = (0...train_x.size_y).map do |y|
	[Matrix.setVectorizedMatrix(train_x[y], 28, 28)]
end

input_size = [28, 28]

layers = [
	InputLayer.new(input_size),
	ConvLayer.new(filter_nb: 8, size: [4, 4], step_y: 1, step_x: 1, lrn: 0.001),
	MaxPoolLayer.new(size: [4, 4], step_x: 4, step_y: 4),
	ConvLayer.new(filter_nb: 4, size: [2, 2], step_y: 2, step_x: 2, lrn: 0.001),
	NetLayer.new(
		size: 10,
	 	activFunction: ActivFunc::SoftMax, 
	 	lrn: 0.001,
		# lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.95),
		# regularizer: Regularizer::L1.new(alpha: 0.01) 
	)
]

net = Cnn.new
net.addLayers(layers)
net.train(
	[train_x, train_y], 
# 	batch_size: 1, 
# 	iteration: 3, 
# 	epoch: 1
)