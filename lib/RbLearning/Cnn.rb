require_relative './ActivFunc'
require_relative './LossFunc'
require_relative './NeuroNet'
require 'benchmark'

class Cnn

	attr_accessor :ann, :layers

	def initialize
		@ann = NeuroNet.new(costFunction: CostFunc::CrossEntropy)
	end

	def forwardProp(arr)
		f_inp = []
		f_out = []
		self.layers.each do |l|
			f_inp.push(arr)
			arr = arr.map do |inputs|
				l.func(inputs)
			end
			f_out.push(arr)
		end
		return [f_inp, f_out]
	end

	def forward_step(train_x, train_y)
		# puts "forwardProp:" 
		# puts Benchmark.measure {self.forwardProp(train_x)}
		f_in, f_out = self.forwardProp(train_x)		
		x = self.transform_to_mat(f_out.last)
		loss = self.ann.calc_loss(x, train_y)
		STDERR.puts "Error: #{loss}"
		zs, act = self.ann.feedForward(x)
		[f_in, f_out, zs, act]
	end

	def backward_step(train_x, train_y, f_in, f_out, zs, act)
		dz = self.ann.update_weigths(zs, act, train_y)
		dz = self.reverse_transform_to_mat(dz, f_out.last, f_out.last.last.last.size_y, f_out.last.last.last.size_x)
		# puts "backProp:" 
		# puts Benchmark.measure {self.backProp(dz, f_in, f_out)}
		self.backProp(dz, f_in, f_out)
	end

	def one_iteration_train(train_x, train_y)
		f_in, f_out, zs, act = forward_step(train_x, train_y)
		backward_step(train_x, train_y, f_in, f_out, zs, act)
	end

	def train(data, batch_size: 32, iteration: 42, epoch: 500)
		train_x, train_y = data
		(0...epoch).each do |ep|
			puts "epoch: #{ep}"
			batch_x, batch_y = Cnn.batch(y: train_y, x: train_x, batch_size: batch_size)
			(0..iteration).each do |i|
				self.one_iteration_train(batch_x, batch_y)
			end
		end
	end

	def self.batch(y: [], x: [], batch_size: 24)
		Random.srand
		indexes = (0...batch_size).map { Random.rand(0...y.size_y) }
		batch_x = indexes.map do |i|
			x[i]
		end
		batch_y = Matrix.set(indexes.map do |i|
			y[i]
		end)

		return [batch_x, batch_y]
	end

	def backProp(dx, f_in, f_out)
		(0...self.layers.size).reverse_each  do |i|
			(0...f_in[i].size).map do |v|
			
				inputs = f_in[i][v]
				l = self.layers[i]
				
				if l.is_a?(MaxPoolLayer)
					dx[v] = l.resize(inputs, dx[v])
				elsif l.is_a?(ConvLayer)
					dw = l.convdw(dx[v], inputs)
					l.updateFilters(dw)
					dx[v] = l.convdx(dx[v], inputs[0].size_y, inputs[0].size_x)
				end
			end
		end
	end

	def reverse_transform_to_mat(m, size, size_y, size_x)
		res = (0...m.size_y).map do |y|
			(0...m.size_x).step(size_y * size_x).map do |x|
				tmp = m.getMat(y, x, 1, size_x * size_y)
				tmp.reshape([size_y, size_x])
				tmp
			end
		end
		return res
	end

	def transform_to_mat(arr)
		x = Matrix.set(
			arr.map do |inputs|
				inputs.map do |inp|
					shape = [1, inp.size_y * inp.size_x]
					tmp = inp.copy
					tmp.reshape(shape)
					tmp.matrix
				end.flatten
			end
		)
		return x
	end

	def addLayers(layers)
		input_layer = layers.first
		layers.shift
		ex_mat = Matrix.new(input_layer.size_y, input_layer.size_x)
		self.layers = layers.select { |e| !e.is_a?(NetLayer) }

		f_in, f_out = self.forwardProp([[ex_mat]])
		
		x = self.transform_to_mat(f_out.last)
		
		ann_layers = [InputLayer.new([x.size_y, x.size_x])]
		ann_layers += layers.select { |e| e.is_a?(NetLayer) }

		self.ann.addLayers(ann_layers)
	end


	private
end

class MaxPoolLayer
	attr_accessor :size_y, :size_x, :step_y, :step_x

	def initialize(size: [4, 4], step_y: 1, step_x: 1)
		self.size_y = size.first
		self.size_x = size.last
		self.step_y = step_y
		self.step_x = step_x
	end

	def func(inputs)
		res = inputs.map do |inp|
			Matrix.set(
				(0...inp.size_y).step(self.step_y).map do |y|
					(0...inp.size_x).step(self.step_x).map do |x|
						inp.getMax(y, x, self.size_y, self.size_x).first
					end
				end
			)
		end
		return res
	end

	def deriv(inputs)
		res = inputs.map do |inp|
			tmp = Matrix.new(inp.size_y, inp.size_x, 0)
			(0...inp.size_y).step(self.step_y).each do |y|
			    (0...inp.size_x).step(self.step_x).each do |x|
					val, pos_y, pos_x = inp.getMax(y, x, self.size_y, self.size_x)
					tmp[pos_y, pos_x] = 1
				end
			end
			tmp
		end
		return res
	end

	def resize(org_inputs, dz)
		res = (0...org_inputs.size).map do |i|
			inp = org_inputs[i];
			m = Matrix.new(inp.size_y, inp.size_x, 0)
			d = dz[i]
			(0...m.size_y).step(step_y) do |y|
				(0...m.size_x).step(step_x) do |x|
					v, pos_y, pos_x = inp.getMax(y, x, self.size_y, self.size_x)
					m[pos_y, pos_x] = d[y / size_y, x / size_x]
				end
			end
			m
		end
		return res
	end

end


class ConvLayer

	attr_accessor :filters, :step_y, :step_x, :f_size_y, :f_size_x, :lrn

	def initialize(filter_nb: 1, size: [4, 4], step_y: 1, step_x: 1, lrn: 0.1, initFunc: lambda { return Random.rand(-0.1...0.1) } )
		self.filters = []
		self.step_y = step_y
		self.step_x = step_x
		self.f_size_y, self.f_size_x = size
		self.lrn = lrn
		(0...filter_nb).each do |i|
			self.filters[i] = initWeightsFunc(size.first, size.last, initFunc)
		end
	end

	def updateFilters(inputs)
		inputs.each_with_index do |inp, i|
			(0...inp.size_y).step(self.step_y) do |y|
				(0...inp.size_x).step(self.step_x) do |x|
					f = self.filters[i % self.filters.size]
					(0...f.size_y).each do |f_y|
						(0...f.size_x).each do |f_x|
							f[f_y, f_x] -= inp[y + f_y, x + f_x] * self.lrn if inp[y + f_y, x + f_x]
						end
					end
				end
			end
		end
	end

	def convdx(dz, size_y, size_x)
		res = []
		dz.each_with_index do |d, i|
			f = self.filters[i % self.filters.size]
			if res[i / self.filters.size].nil?
				res[i / self.filters.size] = applydx(d, f, size_y, size_x)
			else
				res[i / self.filters.size] << applydx(d, f, size_y, size_x)
			end
		end
		return res
	end

	def applydx(d, f, size_y, size_x)
		dx = Matrix.new(size_y, size_x, 0)
		(0...d.size_y).each do |y|
			(0...d.size_x).each do |x|
				(0...f_size_y).each do |f_y|
					(0...f_size_x).each do |f_x|
						dx[y * self.step_y + f_y, x * self.step_x + f_x] += f[f_y, f_x] * d[y, x]
					end
				end
			end
		end
		return dx
	end

	def convdw(dz, inputs)
		res = []
		dz.each_with_index do |d, i|
			inp = inputs[i / self.filters.size]
			res.push(applydw(inp, d))
		end
		return res
	end

	def applydw(inp, d)
		dw = Matrix.new(inp.size_y, inp.size_x, 0)
		(0...d.size_y).each do |y|
			(0...d.size_x).each do |x|
				(0...f_size_y).each do |f_y|
					(0...f_size_x).each do |f_x|
						dw[y * self.step_y + f_y, x * self.step_x + f_x] += inp[y * self.step_y + f_y, x * self.step_x + f_x] * d[y, x]
					end
				end
			end
		end
		return dw
	end

	def func(inputs)
		res = []
		inputs.each do |inp|
			self.filters.each do |f|
				res.push(convolution(inp, f))
			end
		end
		return res
	end

	def convolution(inp, filter)
		size_y = ((inp.size_y / step_y.to_f) - ((filter.size_y - step_y) / step_y.to_f)).floor
		size_x = ((inp.size_x / step_x.to_f) - ((filter.size_x - step_x) / step_x.to_f)).floor
		conv = Matrix.new(size_y, size_x)
		
		(0...conv.size_y).each do |y|
			(0...conv.size_x).each do |x|
				conv[y, x] = applyFilter(inp, filter, y * step_y, x * step_x)
			end
		end
		return conv
	end

	def applyFilter(inp, filter, pos_y = 0, pos_x = 0)
		sum = (0...filter.size_y).sum do |y|
			(0...filter.size_x).sum do |x|
				filter[y, x] * inp[pos_y + y, pos_x + x]
			end
		end
		return sum
	end

	def initWeightsFunc(size_y, size_x = size_y, func = weigthInitFunc)
		w = Matrix.new(size_y, size_x)
		(0...size_y).each do |y|
			(0...size_x).each do |x|
				w[y, x] = func.call()
			end
		end
		return w
	end


end