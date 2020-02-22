require './../lib/RbLearning'

train_x = Matrix.set([[5]])
train_y = Matrix.set([[7]])

layers = [
	NetLayer.new(
		rowSize: 128,
		columnSize: train_x.size_x,
	 	# activFunction: ActivFunc::ReLu, 
	 	lrn: 0.1,
		# lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.95),
		# regularizer: Regularizer::L1.new(alpha: 0.01) 
	)
]

net = NeuroNet.new(lossFunction: LossFunc::MeanSqrtErr)
net.addLayers(layers)
net.train(
	[train_y, train_x], 
	batch_size: 1, 
	iteration: 3, 
	epoch: 1
)