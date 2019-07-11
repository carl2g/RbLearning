require_relative './ActivFunc'
require_relative './LossFunc'

class NeuroNet

	include ActivFunc
	include LossFunc

	attr_accessor :layers, :lossFunc, :lastLoss

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
			x.normalize
			x = l.w * x + l.b
			zs.push(x)
			x = l.activFunc.func(x)
			act.push(x)
		end
		act = dropOut(act)
		return [zs, act]
	end

	def backPropagation(zs, act, y)
		i = @layers.size - 1

		dz =  @lossFunc.deriv(act[i + 1], y) ** @layers[i].activFunc.derivate(zs[i])
		dw = (dz * act[i].transpose).applyOp(:*, @layers[i].lrn / dz.size_x)
		dwOpt, dbOpt = @layers[i].optimize(dw, dz.applyOp(:*,  @layers[i].lrn /  dz.size_x))

		# dw.printShape
		# @layers[i].w.printShape
		# dz.printShape
		# dbOpt.printShape
		# exit
		w = [@layers[i].w - dwOpt]
		b = [@layers[i].b - dbOpt.sumAxis]

		(0...@layers.size - 1).reverse_each do |i|
			
			# puts "============== WANTED ========="
			# @layers[i].w.printShape
			# puts "============== HAVE ========="
			# @layers[i + 1].w.printShape
			# dz.printShape

			tmp = @layers[i + 1].w.transpose * dz 
			
			# tmp.printShape
			# act[i].printShape
			
			dz = tmp ** @layers[i].activFunc.derivate(zs[i])
			dw = (dz * act[i].transpose).applyOp(:*, @layers[i].lrn / dz.size_x)
			dwOpt, dbOpt = @layers[i].optimize(dw, dz.applyOp(:*, @layers[i].lrn / dz.size_x))
			
			w.push(@layers[i].w - dwOpt)
			# @layers[i].b.printShape
			# dbOpt.sumAxis.printShape
			b.push(@layers[i].b - dbOpt.sumAxis)
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
		@lastLoss = @lossFunc.loss(act.last, y)
		STDERR.puts "Error: #{@lastLoss}"
		return @layers
	end

end
