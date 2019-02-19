require_relative './ActivFunc'
require_relative './LossFunc'

class NeuroNet

	include ActivFunc
	include LossFunc

	def initialize

	end

	def feedForward(layers, x)
		act = [x]
		zs = []
		layers.each do |l|
			x = x * l.w + l.b
			zs.push(x)
			x = send(l.actFunc, x)
			act.push(x)
		end
		return [zs, act]
	end


	def backPropagation(zs, act, y, layers, lrn = 0.005)
		i = layers.size - 1

		dz = costFunc(act[i + 1], y) ** send(layers[i].primeActFunc, zs[i])
		dw = [layers[i].w - (act[i].transpose * dz).applyOp(:*, lrn / dz.size_x)]
		db = [layers[i].b - dz.sumOrd.applyOp(:*, lrn / dz.size_x)]

		(0...layers.size - 1).reverse_each do |i|
			tmp = dz * layers[i + 1].w.transpose
			dz = tmp ** send(layers[i].primeActFunc, zs[i])
			dw.push(layers[i].w - (act[i].transpose * dz).applyOp(:*, lrn / dz.size_x))
			db.push(layers[i].b - dz.sumOrd.applyOp(:*, lrn / dz.size_x))
		end
		return [dw.reverse!, db.reverse!]
	end

	def train(layers, x, y, lrn = 0.05, printErr = false)
		zs, act = feedForward(layers, x)
		ws, bs = backPropagation(zs, act, y, layers, lrn)
		(0...layers.size).each do |i|
			layers[i].w = ws[i]
			layers[i].b = bs[i]
		end
		puts "Error: #{meanAbsErr(act.last, y)}" if printErr
		return layers
	end

end
