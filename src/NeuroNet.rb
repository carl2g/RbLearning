require_relative './ActivFunc'
require_relative './LossFunc'

class NeuroNet

	include ActivFunc
	include LossFunc

	def initialize

	end

	def feedForward(layers, x, actFunc = 'sigmoid')
		act = [x]
		zs = []
		layers.each do |l|
			x = x * l.w + l.b
			zs.push(x)
			x = send(actFunc, x)
			act.push(x)
		end
		return [zs, act]
	end

	def backPropagation(zs, act, y, layers, lrn = 0.05, lossFunc = 'sigmoidPrime')
		puts "Error: #{err(act.last, y)}"
		i = layers.size - 1
		dz = costFunc(act.last, y)
		dw = [layers[i].w - act[i].transpose * dz.applyOp(:*, lrn * (1.0 / dz.size_y)) ** send(lossFunc, act[i + 1])]
		db = [layers[i].b - dz.sumAxis]

		(0...layers.size - 1).reverse_each do |i|
			tmp = dz * layers[i + 1].w.transpose
			dz = tmp ** send(lossFunc, act[i + 1])
			dw.push(layers[i].w - act[i].transpose * dz.applyOp(:*, lrn * (1.0 / dz.size_y)))
			db.push(layers[i].b - dz.sumAxis)
		end

		return [dw.reverse!, db.reverse!]
	end

	def train(layers, x, y, lrn = 0.05, actFunc = 'sigmoid', actFuncPrime = actFunc + "Prime")
		zs, act = feedForward(layers, x, actFunc)
		ws, bs = backPropagation(zs, act, y, layers, lrn, actFuncPrime)
		(0...layers.size).each do |i|
			layers[i].w = ws[i]
			layers[i].b = bs[i]
		end
		return layers
	end

end
