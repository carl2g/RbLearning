require './src/ActivFunc'
require './src/LossFunc'

class NeuroNet

	include ActivFunc
	include LossFunc

	def initialize

	end

	def feedForward(ws, x, bs, actFunc = 'sigmoid')
		act = [x]
		zs = []
		(0...ws.size).each do |i|
			x = x * ws[i] + bs[i]
			zs.push(x)
			x = send(actFunc, x)
			act.push(x)
		end
		return [zs, act]
	end

	def backPropagation(zs, act, y, ws, bs, lrn = 0.05, lossFunc = 'sigmoidPrime')
		puts "Error: #{err(act.last, y)}"

		i = ws.size - 1
		dz = costFunc(act.last, y)
		dw = [ws[i] - act[i].transpose * dz.applyOp(:*, lrn)]
		db = [bs[i] - dz.sumAxis]

		(0...ws.size - 1).reverse_each do |i|
			tmp = dz * ws[i + 1].transpose
			dz = tmp ** send(lossFunc, act[i + 1])
			dw += [ws[i] - act[i].transpose * dz.applyOp(:*, lrn)]
			db += [bs[i] - dz.sumAxis]
		end

		return [dw.reverse, bs]
	end

	def train(ws, bs, x, y, actFunc = 'sigmoid')
		zs, act = feedForward(ws, x, bs, actFunc)
		ws, bs = backPropagation(zs, act, y, ws, bs)
		# act.last.printM(3)
		# y.printM(3)
		return [ws, bs]
	end

	def initWeights(size_y, size_x = size_y, min = -1.0, max = 1.0)
		r = Random.new
		w = Matrix.new(size_y, size_x)
		(0...size_y).each do |y|
			(0...size_x).each do |x|
				w[y][x] = r.rand(min...max)
			end
		end
		return w
	end


end
