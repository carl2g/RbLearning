require './../lib/RbLearning'
require 'benchmark'

include Statistic
include Utils
include ActivFunc

train = DataManager.new("./input/train.csv")
data_y = train.remove('Vegetation')
train.remove('Id')

tree = DecisionTree.new(informationGain: Entropy.new)
class_tree = tree.generate_tree(train.data, data_y)
row = hash_select_index(train.data, 1)
puts "classify: #{DecisionTree::classify(row, class_tree)}"
