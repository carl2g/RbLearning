require './../lib/RbLearning'
require 'benchmark'

include Statistic
include Utils
include ActivFunc

dataMTrain = DataManager.new("./input/train.csv")
dataMTrain.remove('Id')

dataMTest = DataManager.new("./input/test.csv")
dataMTest.remove('Id')
dataMTest.addDumies

# ======== Mode is NA =======
dataMTrain.remove('LotFrontage')
# ======== Only one value =======
dataMTrain.remove('Street')
# ======== Mode is NA =======
dataMTrain.remove('Alley')
# ======== Only one value =======
dataMTrain.remove('Utilities')
# ======== Mode is NA =======
dataMTrain.remove('CentralAir')
# ======== Only one value =======
dataMTrain.remove('MiscVal')
# ======== Mode is NA =======
dataMTrain.remove('MiscFeature')
# ======== Mode is NA =======
dataMTrain.remove('Fence')
# ======== Mode is NA =======
dataMTrain.remove('PoolQC')
# ======== Only one value =======
dataMTrain.remove('PoolArea')
# ======== Only one value =======
dataMTrain.remove('ScreenPorch')
# ======== Mode is NA =======
dataMTrain.remove('GarageCond')
# ======== Mode is NA =======
dataMTrain.remove('FireplaceQu')

dataMTrain.replace_val_label("GarageYrBlt", "NA", mean(dataMTrain["GarageYrBlt"]).round)
dataMTrain.replace_val_label("MasVnrArea", "NA", mean(dataMTrain["MasVnrArea"]).round)
dataMTrain.removeRows("NA")

data_y = dataMTrain.remove('SalePrice')

# data_y = data_y.map do |v|
# 	Math.log(v.to_f)
# end

data_y = Matrix.setVectorizedMatrix(data_y, data_y.size, 1)

# puts dataMTrain.labels
# exit
quant_labs = [
	"OverallQual", 
	"GrLivArea",
	"GarageArea", 
	"GarageCars",
	"1stFlrSF",
	"MasVnrArea"
]

qual_labs= [
	"MSZoning",
	"LotShape",
	"LandContour",
	"LotConfig",
	"LandSlope",
	# "Neighborhood",
	"Condition1",
	"Condition2",
	"BldgType",
	"HouseStyle",
	"RoofMatl",
	# "Exterior1st",
	# "Exterior2nd",
	"MasVnrType",
	"MasVnrArea",
	"ExterQual",
	"ExterCond",
	"Foundation",
	"BsmtQual",
	"BsmtCond",
	"BsmtExposure",
	"BsmtFinType1",
	"BsmtFinType2",
	"Heating",
	"HeatingQC",
	"Electrical",
	"KitchenQual",
	"Functional",
	"GarageType",
	"GarageYrBlt",
	"GarageFinish",
	"GarageQual",
	"Fence",
	"SaleType",
	"SaleCondition"
]

quant_labs.each_with_index do |lab, i|
	q1 = Statistic::quartile(dataMTrain[lab], 1)
	q3 = Statistic::quartile(dataMTrain[lab], 3)
	iq = q3.to_f - q1.to_f
	dataMTrain[lab].each do |v|
		if v.to_f < q1 - (1.5 * iq) || v.to_f > q3 + (1.5 * iq)
			# puts "#{lab} #{v}"
			dataMTrain.removeRow(lab, v)
		end
	end
end

dataMTrain.keepLabels(quant_labs + qual_labs)
dataMTrain.addDumies

dataMTrain.keepLabels(dataMTest.labels)
dataMTest.keepLabels(dataMTrain.labels)

# puts dataMTrain.labels.size
# exit

# data = dataMTrain.matrix << data_y
# puts (labs << "SalePrice").join(",")
# puts data.to_csv
# # exit
# data = dataMTrain.matrix

train_x, train_y, validation_x, validation_y = dataMTrain.split([0.9, 0.1], data_y, random: true)
 

train_x = Statistic::standerdized(train_x, axis: 1)
validation_x = Statistic::standerdized(validation_x, axis: 1)

# validation_x.printM
# exit

layers = [
	Input.new(train_x.size_x),
	NetLayer.new(
		size: 32,
	 	# activFunction: ActivFunc::ReLu, 
	 	lrn: 0.00000001,
		# lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.95),
		# regularizer: Regularizer::L1.new(alpha: 0.01) 
	),
	NetLayer.new( 
		size: 1,
	 	# activFunction: ActivFunc::ReLu,
	 	lrn: 0.00000001, 
		# lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.95),
		# regularizer: Regularizer::L1.new(alpha: 0.01)
	)
	# ,
	# NetLayer.new( 
	# 	size: 1,
	#  	activFunction: ActivFunc::ReLu,
	#  	lrn: 0.001, 
	# 	# lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.95),
	# 	# regularizer: Regularizer::L1.new(alpha: 0.01)
	# )
]

net = NeuroNet.new(costFunction: CostFunc::MeanSqrtErr)
net.addLayers(layers)
net.train(
	[train_y, train_x], 
	batch_size: 32, 
	iteration: 24000,
	epoch: 1
)

net.layers.last.w.printM(3)

pred = net.pred(validation_x)

pred.matrix = pred.matrix.map {|v| Math.exp(v.to_f)}
validation_y.matrix = validation_y.matrix.map {|v| Math.exp(v.to_f)}

(0...10).each do |i|
	puts "#{pred[i, 0].round}  == #{validation_y[i, 0].round}"
end

puts LossFunc::MeanSqrtErr::loss(pred, validation_y)

data_x = Statistic::standerdized(dataMTest.matrix)
pred = net.pred(data_x)

CSV.open("./res.csv", "wb") do |csv|
	csv << ['Id', 'SalePrice']
	(0...pred.size_y).each do |i|
		csv <<  [1461 + i, Math.exp(pred[i, 0])]
	end
end