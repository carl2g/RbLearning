require './../lib/RbLearning'
require 'benchmark'

include Statistics
include Utils
include ActivFunc
include LossFunc::CrossEntropy

y = Matrix.set([
	[1, 0, 0]
])

x = Matrix.set([
	# [0.4, 0.3, 0.05, 0.05, 0.2]
	[0.2698, 0.3223, 0.4078]
])

res = LossFunc::CrossEntropy.func(x, y)

res.printM(3)

delt = LossFunc::CrossEntropy.derivate(x, y)

delt.printM(3)

ActivFunc::SoftMax.derivate(delt).printM(3)