require_relative './ActivFunc'
require_relative './LossFunc'
require_relative './DataManager'

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
			w = l.w
			x = (x * w.transpose) + l.b
			zs.push(x)
			x = l.activFunc.func(x)
			act.push(x)
		end
		return [zs, act]
	end

	def pred(data_x)
		zs, acts = feedForward(data_x)
		return acts.last
	end

	def backPropagation(zs, act, y)
		i = @layers.size - 1

		loss = @lossFunc.deriv(act[i + 1], y)
		# loss = @layers[i].regularizer.regularizeBackward(loss, @layers[i].w)
		dz = loss ** @layers[i].activFunc.derivate(zs[i])
		dw = dz.transpose * act[i]

		dwOpt, dbOpt = @layers[i].optimize(dw.applyOp(:*, @layers[i].lrn / dz.size_x), dz.applyOp(:*, @layers[i].lrn / dz.size_x))
		
		w = [@layers[i].w - dwOpt]
		b = [@layers[i].b - dbOpt.sumOrd.transpose]

		(0...@layers.size - 1).reverse_each do |i|
			
			tmp = dz * @layers[i + 1].w
			# dz = @layers[i].regularizer.regularizeBackward(dz, @layers[i].w)
			dz = tmp ** @layers[i].activFunc.derivate(zs[i])
			dw = dz.transpose * act[i]
			dwOpt, dbOpt = @layers[i].optimize(dw.applyOp(:*, @layers[i].lrn / dz.size_x), dz.applyOp(:*, @layers[i].lrn / dz.size_x))
			
			w.push(@layers[i].w - dwOpt)
			b.push(@layers[i].b - dbOpt.sumOrd.transpose)
		end
		return [w.reverse!, b.reverse!]
	end

	def train(data, batch_size: 32, iteration: 42, epoch: 500)
		data_y, data_x = data

		(0...epoch).each do |ep|
			batch_x, batch_y = DataManager.batch(y: data_y, x: data_x, batch_size: batch_size)
			(0..iteration).each do |i|

				layers, loss = internal_train(batch_y, batch_x)
				
				STDERR.puts "Error: #{loss}"
				puts "epoch: #{ep} iteration: #{i}"
				puts "=" * 30
			end
		end
		return layers
	end

	private
		def internal_train(y, x)
			zs, act = feedForward(x)
			ws, bs = backPropagation(zs, act, y)
			
			(0...@layers.size).each do |i|
				@layers[i].w = ws[i]
				@layers[i].b = bs[i]
			end
			
			loss = @lossFunc.loss(act.last, y)

			return [@layers, loss]
		end

end
