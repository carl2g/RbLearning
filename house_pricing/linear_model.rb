require './../lib/RbLearning'
require 'benchmark'

include Statistic
include Utils
include ActivFunc

dataMTrain = DataManager.new("./input/train.csv")
dataMTrain.remove('Id')
data_y = dataMTrain.remove('SalePrice')
data_y.map!(&:to_f)

min = data_y.min
max = data_y.max

# data_y = data_y.map {|v| 1 + (v.to_f - min) / (max - min)}
# data_y = data_y.map {|v| Math.log(v)}


data_y = Matrix.setVectorizedMatrix(data_y, data_y.length, 1)

dataMTrain.replace_val_label("GarageYrBlt", "NA", mean(dataMTrain["GarageYrBlt"]).round)
dataMTest = DataManager.new("./input/test.csv")
dataMTest.remove('Id')
dataMTest.replace_val_label("GarageYrBlt", "NA", mean(dataMTrain["GarageYrBlt"]).round)

nn = NeuroNet.new

data = dataMTrain.matrix
tr_data = data.transpose

# (0...tr_data.size_y).each do |i|
# 	lab = dataMTrain.labels[i]
# 	puts "#{lab} #{Statistic::correlation(tr_data[i], data_y.matrix).round(3)}"
# end
# exit

lab = "GarageCars"
puts "#{lab} #{Statistic::correlation(dataMTrain[lab], data_y.matrix).round(3)}"

# Correlation: 0.714
dataMTrain["GrLivArea"].each_with_index do |v, i|
	if v.to_f > 4000
		dataMTrain.removeAt(i)
		data_y.drop_row(i)
	end
end
# Correlation: 0.727


# Correlation: 0.61
dataMTrain["1stFlrSF"].each_with_index do |v, i|
	if v.to_f > 4500
		dataMTrain.removeAt(i)
		data_y.drop_row(i)
	end
end
# Correlation: 0.628

# Correlation: 0.658
# dataMTrain["GarageCars"].each_with_index do |v, i|
# 	if v.to_f > 3.5
# 		dataMTrain.removeAt(i)
# 		data_y.drop_row(i)
# 	end
# end
# Correlation: 0.672

puts "#{lab} #{Statistic::correlation(dataMTrain[lab], data_y.matrix).round(3)}"
dataMTrain.addDumie("GarageCars")

dataMTrain["MSSubClass"] = dataMTrain["MSSubClass"].map do |v|
	if v.to_f <= 60
		1
	elsif v.to_f > 140
		3
	else
		2
	end 
end


# dataMTrain.addDumies
# labs = dataMTrain.labels
labs = [
	"OverallQual", 
	# "LotArea",
	"GrLivArea",
	# "YearBuilt", 
	# "GarageYrBlt", 
	# "MSSubClass", 
	# "TotRmsAbvGrd", 
	"GarageArea", 
	# "GarageCars_0", 
	# "GarageCars_1", 
	# "GarageCars_2", 
	# "GarageCars_3", 
	# "GarageCars_4", 
	# "Fireplaces",
	# "WoodDeckSF",
	# "OpenPorchSF",
	# "HalfBath",
	# "FullBath",
	# "BsmtFullBath",
	"1stFlrSF",
	# "2ndFlrSF",
	# "BsmtFinSF1",
	# "MasVnrArea",
	# "YearRemodAdd"
]

dataMTrain.keepLabels(labs)

data = dataMTrain.matrix << data_y
puts (labs << "SalePrice").join(",")
puts data.to_csv
# exit
data = dataMTrain.matrix

# exit

tr_data = data.transpose

puts "=" * 100
(0...tr_data.size_y).each do |i|
	lab = dataMTrain.labels[i]
	puts "#{lab} #{Statistic::correlation(tr_data[i], data_y.matrix).round(3)}"
	# puts "#{dataMTrain[lab].uniq}"
end

train_x, train_y, validation_x, validation_y = dataMTrain.split([0.9, 0.1], data_y)
 
data_x = train_x
validation_x = validation_x
data_y = Matrix.setVectorizedMatrix(train_y.matrix, train_y.size_y, 1)

coefs = poylnomialRegression(data_x, data_y)
pred = validation_x * coefs
# exit

# data_x.printM
# validation_x.printM

coefs.printM(6)

# pred.matrix = pred.matrix.map {|v| CMath.exp(1)**(v.to_f)}
# pred.matrix = pred.matrix.map {|v| (v.to_f - 1) * (max - min) + min}
# validation_y.matrix = validation_y.matrix.map {|v| CMath.exp(1)**(v.to_f)}
# validation_y.matrix = validation_y.matrix.map {|v| (v.to_f - 1) * (max - min) + min}

(0...10).each do |i|
	puts "#{pred[i, 0].round}  == #{validation_y[i, 0].round}"
end

# pred.matrix = pred.matrix.map {|v| (Math.exp(v + 1) + pred.matrix.min) * (pred.matrix.max - pred.matrix.min) }
# validation_y.matrix = validation_y.matrix.map {|v| Math.exp(v + 1)}

puts LossFunc::MeanSqrtErr::loss(pred, validation_y)

# data_x = Statistic::normalize_range(dataMTest.matrix, axis: 1)
# data_x = data_x << Matrix.new(data_x.size_y, 1, 1)

# pred = data_x * coefs

CSV.open("./res.csv", "wb") do |csv|
	csv << ['Id', 'SalePrice']
	(0...pred.size_y).each do |i|
		csv <<  [1461 + i, pred[i, 0]]
	end
end
