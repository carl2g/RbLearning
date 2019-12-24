require './../lib/RbLearning'
require 'benchmark'

include Statistic
include Utils
include ActivFunc

dataMTrain = DataManager.new("./input/train.csv")
dataMTrain.remove('Id')
data_y = dataMTrain.remove('SalePrice')
data_y = Matrix.setVectorizedMatrix(data_y, data_y.size, 1)

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

dataMTrain.replace_val_label("GarageYrBlt", "NA", mean(dataMTrain["GarageYrBlt"]).round)

# ============== GrLivArea ==============
# Correlation: 0.709
dataMTrain["GrLivArea"].each_with_index do |v, i|
	if v.to_f > 4000
		dataMTrain.removeAt(i)
		data_y.drop_row(i)
	end
end
# Correlation: 0.721


# ============== 1stFlrSF ==============
# Correlation: 0.606
dataMTrain["1stFlrSF"].each_with_index do |v, i|
	if v.to_f > 4500
		dataMTrain.removeAt(i)
		data_y.drop_row(i)
	end
end
# Correlation: 0.625


# ============== GarageCars ==============
dataMTrain.addDumie("GarageCars")


# ============== MSSubClass ==============
# Correlation: -0.088
dataMTrain["MSSubClass"] = dataMTrain["MSSubClass"].map do |v|
	if v.to_f <= 60
		1
	elsif v.to_f > 140
		3
	else
		2
	end 
end
# Correlation: -0.182

dataMTrain.addDumie("MSSubClass")

dataMTrain.addDumies
labs = dataMTrain.labels
# labs = [
# 	"OverallQual", 
# 	# "LotArea",
# 	"GrLivArea",
# 	"YearBuilt", 
# 	# "GarageYrBlt", 
# 	"MSSubClass_1", 
# 	"MSSubClass_2", 
# 	"MSSubClass_3", 
# 	"TotRmsAbvGrd", 
# 	"GarageArea", 
# 	"GarageCars_0", 
# 	"GarageCars_1", 
# 	"GarageCars_2", 
# 	"GarageCars_3", 
# 	"GarageCars_4", 
# 	# "Fireplaces",
# 	# "WoodDeckSF",
# 	# "OpenPorchSF",
# 	# "HalfBath",
# 	# "FullBath",
# 	# "BsmtFullBath",
# 	"1stFlrSF",
# 	# "2ndFlrSF",
# 	# "BsmtFinSF1",
# 	# "MasVnrArea",
# 	"YearRemodAdd"
# ]

data = dataMTrain.matrix
tr_data = data.transpose
keep_labs = []
labs.each_with_index do |lab, i|
	# puts "#{lab} #{Statistic::correlation(tr_data[i], data_y.matrix).round(3)}"
	# puts "#{dataMTrain[lab].uniq}"
	# if Statistic::correlation(dataMTrain[lab], data_y.matrix).round(3).abs > 0
	# 	puts "#{lab} #{Statistic::correlation(dataMTrain[lab], data_y.matrix).round(3).abs}"
		keep_labs << lab
	# end
end
# exit

dataMTrain.keepLabels(keep_labs)

# data = dataMTrain.matrix << data_y
# puts (labs << "SalePrice").join(",")
# puts data.to_csv
# # exit
# data = dataMTrain.matrix

train_x, train_y, validation_x, validation_y = dataMTrain.split([0.9, 0.1], data_y)
 
data_x = Statistic::normalize_range(train_x, axis: 1)
validation_x = Statistic::normalize_range(validation_x, axis: 1)
data_y = train_y

layers = [
	NetLayer.new(
	 	256, 
	 	data_x.size_x, 
	 	ActivFunc::ReLu, 
	 	lrn: 0.00000005, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
		# regularizer: Regularizer::L1.new(alpha: 0.1)
	),
	NetLayer.new(
	 	64, 
	 	256, 
	 	ActivFunc::ReLu,
	 	lrn: 0.000000005, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
	),
	NetLayer.new(
	 	1, 
	 	64, 
	 	ActivFunc::ReLu, 
	 	lrn: 0.000000005, 
		lrnOptimizer: LrnOptimizer::Momentum.new(beta: 0.99),
	)
]

net = NeuroNet.new(lossFunction: LossFunc::MeanSqrtErr)
net.addLayers(layers)
net.train([data_y, data_x], batch_size: 8, iteration: 24, epoch: 50)
# net.layers.first.w.printM

pred = net.pred(validation_x)

(0...10).each do |i|
	puts "#{pred[i, 0].round}  == #{validation_y[i, 0].round}"
end

puts LossFunc::MeanSqrtErr::loss(pred, validation_y)

