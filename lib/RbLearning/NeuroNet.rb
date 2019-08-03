require_relative './ActivFunc'
require_relative './LossFunc'

class NeuroNet

	include ActivFunc
	include LossFunc

	attr_accessor :layers, :lossFunc, :lastLoss

	def initialize(lossFunction: LossFunc::MeanSqrtErr)
		@layers = []
		@lossFunc = lossFunction
	end

	def addLossFunc(func)
		@lossFunc = func
	end

	def compile
		@layers.each do |l|

		end
	end

	def addLayer(newLayer)
		@layers << newLayer
	end

	def addLayers(layers)
		@layers = layers
	end

	def feedForward(x)
		act = [x]
		zs = []
		@layers.each do |l|
			w = l.regularizeForward(l.w)
			x = w * x + l.b
			zs.push(x)
			x = l.activFunc.func(x)
			act.push(x)
		end
		return [zs, act]
	end

	def predict(x)
		res = nil
		@layers.each do |l|
			x = l.w * x + l.b
			x = l.activFunc.func(x)
			res = x
		end
		return res
	end

	def backPropagation(zs, act, y)
		i = @layers.size - 1

		regularize = @layers[i].regularizeBackward(@lossFunc.deriv(act[i + 1], y), @layers[i].w)
		dz = regularize ** @layers[i].activFunc.derivate(zs[i])
		dw = dz * act[i].transpose
		dwOpt, dbOpt = @layers[i].optimize(dw, dz)

		w = [@layers[i].w - dwOpt.applyOp(:*, @layers[i].lrn / dz.size_x)]
		b = [@layers[i].b - dbOpt.sumAxis.applyOp(:*, @layers[i].lrn / dz.size_x)]

		(0...@layers.size - 1).reverse_each do |i|
			
			tmp = @layers[i + 1].w.transpose * dz
			regularize = @layers[i].regularizeBackward(tmp, @layers[i].w)
			dz = regularize ** @layers[i].activFunc.derivate(zs[i])
			dw = dz * act[i].transpose
			dwOpt, dbOpt = @layers[i].optimize(dw, dz)
			
			w.push(@layers[i].w - dwOpt.applyOp(:*, @layers[i].lrn / dz.size_x))
			b.push(@layers[i].b - dbOpt.sumAxis.applyOp(:*, @layers[i].lrn / dz.size_x))
		end
		return [w.reverse!, b.reverse!]
	end

	def train(x, y)
		zs, act = feedForward(x)
		ws, bs = backPropagation(zs, act, y)
		(0...@layers.size).each do |i|
			@layers[i].w = ws[i]
			@layers[i].b = bs[i]
		end
		
		pred = predict(x)
		@lastLoss = @lossFunc.loss(pred, y)
		STDERR.puts "Error: #{@lastLoss}"

		return @layers
	end

end
