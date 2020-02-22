require './../lib/RbLearning'
require 'benchmark'

include Statistic
include Utils
include ActivFunc

dataMTrain = DataManager.new("./input/train.csv")
dataMTrain.remove('Id')

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
data_y.map!(&:to_f)


data_y = data_y.map do |v|
	Math.log(v, Math.exp(1))
end

min = data_y.min
max = data_y.max
std_dev = std_dev(data_y)

# data_y = data_y.map {|v| (v.to_f) }
data_y = Matrix.setVectorizedMatrix(data_y, data_y.length, 1)


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

labs = [
	"OverallQual",
	# "GrLivArea",
	# "GarageArea", 
	# "GarageCars", 
	# "FullBath",
	# "1stFlrSF", 
	# "TotalBsmtSF"
]

# dataMTrain.addDumies

# labs = dataMTrain.labels
keep_labs = []
data = dataMTrain.matrix
tr_data = data.transpose
# labs = dataMTrain.labels

labs.each do |lab|
	keep_labs << lab
	puts "#{lab} #{Statistic::correlation(dataMTrain[lab], data_y.matrix).round(3).abs} #{Statistic::skewness(dataMTrain[lab])}"
end


dataMTrain["GrLivArea"].map! { |v| Math.log(v.to_f, Math.exp(1)) }
lab = "GrLivArea"
puts "#{lab} #{Statistic::correlation(dataMTrain[lab], data_y.matrix).round(3).abs} #{Statistic::skewness(dataMTrain[lab])}"

dataMTrain.keepLabels(keep_labs)


# exit
# puts "#{keep_labs}"
# dataMTrain.keepLabels(keep_labs)
# # exit
# data = dataMTrain.matrix << data_y
# puts (labs << "SalePrice").join(",")
# puts data.to_csv
# exit

train_x, train_y, validation_x, validation_y = dataMTrain.split([0.95, 0.05], data_y, random: true)

new_size = 2
train_x = Matrix.setVectorizedMatrix(train_x.getMat(0, 0, new_size, train_x.size_x).matrix, new_size, train_x.size_x)
train_y = Matrix.setVectorizedMatrix(train_y.getMat(0, 0, new_size, train_y.size_x).matrix, new_size, train_y.size_y)

train_x = Matrix.new(train_x.size_y, 1, 1) << train_x
validation_x = Matrix.new(validation_x.size_y, 1, 1) << validation_x
train_y = Matrix.setVectorizedMatrix(train_y.matrix, train_y.size_y, 1)

coefs = poylnomialRegression(train_x, train_y)
pred = train_x * coefs

puts "=" * 10
puts "coefitiences:" 
coefs.printM(6)
puts "=" * 10

pred.matrix = pred.matrix.map {|v| Math.exp(v.to_f)}
validation_y.matrix = validation_y.matrix.map {|v| Math.exp(v.to_f)}

# (0...pred.size_y).each do |i|
# 	puts "#{pred[i, 0].round(4)}  == #{validation_y[i, 0].round(4)}"
# end

puts "=" * 10
puts LossFunc::MeanSqrtErr::loss(pred, validation_y)

