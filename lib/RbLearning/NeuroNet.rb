require_relative './ActivFunc'
require_relative './LossFunc'
require_relative './CostFunc'

class NeuroNet

	include ActivFunc
	include LossFunc
	include CostFunc

	attr_accessor :layers, :lossFunc

	def initialize
		@layers = []
		@lossFunc = LossFunc::MeanSqrtErr
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

	def dropOut(act)
		r = Random.new
		(0...@layers.size).each do |i|
			size = ((@layers[i].dropOut / 100) * act[i].size_y * act[i].size_x).round
			(0...size).each do |drop_count| 
				y = r.rand(0...act[i].size_y)
				x = r.rand(0...act[i].size_x)
				act[i][y, x] = 0
			end
		end
		return act
	end

	def feedForward(x)
		act = [x]
		zs = []
		@layers.each do |l|
			x.normalize(axis: 1)
			x = x * l.w + l.b
			zs.push(x)
			x = l.activFunc.func(x)
			act.push(x)
		end
		act = dropOut(act)
		return [zs, act]
	end

	def backPropagation(zs, act, y)
		i = @layers.size - 1

		dz = costFunc(act[i + 1], y) ** @layers[i].activFunc.derivate(zs[i])
		dw = (act[i].transpose * dz).applyOp(:*, @layers[i].lrn)
		dwOpt, dbOpt = @layers[i].optimize(dw, dz.applyOp(:*, @layers[i].lrn))

		w = [@layers[i].w - dwOpt]
		b = [@layers[i].b - dz.sumOrd.transpose.applyOp(:*, @layers[i].lrn)]

		(0...@layers.size - 1).reverse_each do |i|
			tmp = dz * @layers[i + 1].w.transpose
			dz = tmp ** @layers[i].activFunc.derivate(zs[i])
			dw = (act[i].transpose * dz).applyOp(:*, @layers[i].lrn)

			dwOpt, dbOpt = @layers[i].optimize(dw, dz.applyOp(:*, @layers[i].lrn))
			
			w.push(@layers[i].w - dwOpt.applyOp(:*, @layers[i].lrn))
			b.push(@layers[i].b - dz.sumOrd.transpose.applyOp(:*, @layers[i].lrn))
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
		STDERR.puts "Error: #{@lossFunc.func(act.last, y)}"
		return @layers
	end

end
