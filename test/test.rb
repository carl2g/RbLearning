require './../lib/RbLearning'
require 'benchmark'

include Statistic
include Utils
include ActivFunc

train = DataManager.new("./input/train.csv")
data_y = train.remove('Survived')
train.remove('PassengerId')
train.removeKeysNullVall(80)
# train.addDumies

tests = DataManager.new("./input/test.csv")
tests.remove('PassengerId')
tests.removeKeysNullVall(80)
# tests.addDumies

tr_data = train.matrix.transpose
remove_lab = []

# (0...tr_data.size_y).each do |i|
# 	puts "#{train.labels[i]} #{Statistic::corelation(tr_data[i], data_y).round(3)}"

# 	if Statistic::corelation(tr_data[i], data_y).round(3).abs < 0.15
# 		remove_lab << train.labels[i]
# 	end
# end

# remove_lab.each do |lab|
# 	train.remove(lab)
# end

train["Name"] = train["Name"].map {|n| n.split(',')[1].split('.')[0].strip}
tests["Name"] = tests["Name"].map {|n| n.split(',')[1].split('.')[0].strip}

train["Age"] = train["Age"].map do |n| 
	n = n.to_i
	if n < 18
		"child"
	elsif n < 40
		"midel_age"
	elsif n < 65
		"advence_age"
	else
		"old"
	end
end

tests["Age"] = tests["Age"].map do |n|
	n = n.to_i
	if n < 18
		"child"
	elsif n < 40
		"midel_age"
	elsif n < 65
		"advence_age"
	else
		"old"
	end
end

train["Fare"] = train["Fare"].map do |n| 
	n = n.to_i
	if n < 20
		"chipe"
	elsif n < 40
		"midle"
	elsif n < 80
		"expensive"
	else 
		"very_expensive"
	end
end

tests["Fare"] = tests["Fare"].map do |n|
	n = n.to_i
	if n < 20
		"chipe"
	elsif n < 40
		"midle"
	elsif n < 80
		"expensive"
	else 
		"very_expensive"
	end
end

# puts train["Fare"]

train.keepLabels(["Sex", "Pclass", "SibSp", "Embarked", "Age", "Name", "Embarked", "Fare"])

train.keepLabels(tests.labels)
tests.keepLabels(train.labels)

tests["Name"][414] = "Mrs"
# exit

tree = DecisionTree.new(informationGain: Entropy.new)
class_tree = tree.generate_tree(train.data, data_y)
puts "Final tree: #{class_tree}"



CSV.open("./res.csv", "wb") do |csv|
	csv << ['PassengerId', 'Survived']
	(0...tests.size_y).each do |i|
		d = hash_select_index(tests.data, i)
		# puts "#{i}: #{d}"
		puts DecisionTree::classify(d, class_tree)
		csv <<  [892 + i, DecisionTree::classify(d, class_tree)]
	end
end