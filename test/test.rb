require './../lib/RbLearning'
require 'benchmark'

include Statistic
include Utils
include ActivFunc
include Entropy

train = DataManager.new("./input/train.csv")
data_y = train.remove('Vegetation')
train.remove('Id')

tree = DecisionTree.new(informationGain: Entropy)
tree.generate_tree(train.data, data_y);