require_relative './ActivFunc'
require_relative './LossFunc'

class NeuroNet

	include ActivFunc
	include LossFunc

	attr_accessor :layers

	def initialize
		@layers = []
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
		dw = [@layers[i].w - (act[i].transpose * dz.applyOp(:*, @layers[i].lrn / dz.size_x))]
		db = [@layers[i].b - dz.sumOrd.transpose.applyOp(:*, @layers[i].lrn / dz.size_x)]

		(0...@layers.size - 1).reverse_each do |i|
			tmp = dz * @layers[i + 1].w.transpose
			dz = tmp ** @layers[i].activFunc.derivate(zs[i])
			dw.push(@layers[i].w - (act[i].transpose * dz.applyOp(:*, @layers[i].lrn / dz.size_x)))
			db.push(@layers[i].b - dz.sumOrd.transpose.applyOp(:*, @layers[i].lrn / dz.size_x))
		end
		return [dw.reverse!, db.reverse!]
	end

	def train(x, y)
		zs, act = feedForward(x)
		ws, bs = backPropagation(zs, act, y)
		(0...@layers.size).each do |i|
			@layers[i].w = ws[i]
			@layers[i].b = bs[i]
		end
		STDERR.puts "Error: #{meanSqrtErr(act.last, y)}"
		return @layers
	end

end
