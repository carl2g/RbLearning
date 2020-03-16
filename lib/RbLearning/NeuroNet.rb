require_relative './ActivFunc'
require_relative './CostFunc'
require_relative './DataManager'

class NeuroNet

	include ActivFunc
	include CostFunc

	attr_accessor :layers, :costFunc, :lastLoss

	def initialize(costFunction: costFunc::MeanSqrtErr)
		@layers = []
		@costFunc = costFunction
	end

	def addcostFunc(func)
		@costFunc = func
	end

	def addLayers(layers)
		input_size = layers.first.size_x
		layers.shift
		@layers = layers
		@layers.each do |l|
			l.initWeigths(l.size, input_size)
			input_size = l.size
		end
	end

	def feedForward(x)
		act = [x]
		zs = []

		@layers.each do |l|
			w = l.w
			b = l.b

			###################################################
			#### Multiplie each X (inputs) with W (weghts) ##########
			#### X(j)0 * W(i)0 + X(j)1 * W(i)1 ... X(j)n * W(i)n ####
			#### Matrice multiplication #############################
			################################
			x = (x * w.transpose) + b

			########################################################
			#### Save input of activation function for derivate ####
			########################################################
			zs.push(x)

			#####################################
			#### Compute activation function ####
			#####################################
			x = l.activFunc.nil? ? x : l.activFunc.func(x)
			#########################################################
			#### Save output of activation function for derivate ####
			#########################################################
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

		####################################################################
		#### Compute the derivate of the cost function and get the loss ####
		####################################################################
		loss = @costFunc.deriv(act[i + 1], y)

		############################################################################################
		#### Compute the regularization L1 / L2, if no regularization set loss will be returned ####
		############################################################################################
		regLoss = @layers[i].regularizer.backward(loss, @layers[i].w)
		
		#####################################################
		#### Compute the activation function or use loss ####
		#####################################################
		if @layers[i].activFunc.nil?
			dz = regLoss
		else
			dz = regLoss ** @layers[i].activFunc.derivate(zs[i])
		end
		
		#################################################
		#### dCost * d(X * Wt) if no activation func ########################
		#### dCost ** dActFunc * d(X * Wt) if the activation func was set ####
		#### d(X * Wt) / d(Wt) = X ##########################################
		#### dCost ** dActFunc * X ####
		###############################
		dw = dz.transpose * act[i]

		##########################################
		#### Apply learning rate on derivates ####
		##########################################
		dw = dw.applyOp(:*, @layers[i].lrn)
		db = dz.sumOrd.applyOp(:*, @layers[i].lrn)

		############################################
		#### Changing derivative for next layer ####
		############################################
		dz = dz * @layers[i].w
		
		#################################################
		#### Compute the learning optimizer function ####
		#################################################
		dw = @layers[i].optimize(dw)

		######################################################
		#### Update weights with the calculated derivates ####
		######################################################
		ws = [@layers[i].w - dw]
		bs = [@layers[i].b - db]
		
		(0...@layers.size - 1).reverse_each do |i|
			
			############################################################################################
			#### Compute the regularization L1 / L2, if no regularization set loss will be returned ####
			############################################################################################
			regLoss = @layers[i].regularizer.backward(dz, @layers[i].w)
			
			#####################################################
			#### Compute the activation function or use loss ####
			#####################################################
			if @layers[i].activFunc.nil?
				dz = regLoss
			else
				dz = regLoss ** @layers[i].activFunc.derivate(zs[i])
			end
			
			#################################################
			#### dCost * d(X * Wt) if no activation func ########################
			#### dCost ** dActFunc * d(X * Wt) if the activation func was set ####
			#### d(X * Wt) / d(Wt) = X ##########################################
			#### dCost ** dActFunc * X ####
			###############################
			dw = dz.transpose * act[i]

			##########################################
			#### Apply learning rate on derivates ####
			##########################################
			dw = dw.applyOp(:*, @layers[i].lrn)
			db = dz.sumOrd.applyOp(:*, @layers[i].lrn)
			
			#################################################
			#### Compute the learning optimizer function ####
			#################################################
			dw = @layers[i].optimize(dw)

			######################################################
			#### Update weights with the calculated derivates ####
			######################################################
			ws.push(@layers[i].w - dw)
			bs.push(@layers[i].b - db)
		end
		return [ws.reverse!, bs.reverse!, dz]
	end

	def calc_loss(x, y)
		zs, act = feedForward(x)
		loss = @costFunc.func(act.last, y)
		loss = @layers.last.regularizer.forward(loss, @layers.last.w)
		loss = loss.sumOrd.applyOp(:/, loss.size_y).sumAxis[0, 0]
		return loss
	end

	def train(data, batch_size: 32, iteration: 42, epoch: 500)
		data_y, data_x = data

		(0...epoch).each do |ep|
			batch_x, batch_y = DataManager.batch(y: data_y, x: data_x, batch_size: batch_size)
			(0..iteration).each do |i|
				internal_train(batch_y, batch_x)
				loss = calc_loss(batch_x, batch_y)
				STDERR.puts "Error: #{loss}"
				puts "epoch: #{ep} iteration: #{i}"
				puts "=" * 30
			end
		end
		return layers
	end

	def update_weigths(zs, act, y)
		ws, bs, dz = backPropagation(zs, act, y)

		(0...@layers.size).each do |i|
			@layers[i].w = ws[i]
			@layers[i].b = bs[i]
		end
		return dz
	end

	# private
		def internal_train(y, x)
			zs, act = feedForward(x)
			self.update_weigths(zs, act, y)
		end

end
