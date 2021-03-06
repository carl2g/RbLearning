require_relative './ActivFunc'
require_relative './LossFunc'
require_relative './Utils'

class DecisionTree

	include Statistic
	include Utils
	attr_accessor :informationGain

	def initialize(informationGain: Entropy.new)
		@informationGain = informationGain
	end

	def select_most_information_gain_feature(data_x, data_y)
		information_gain = 0
		selected_key = nil
		selected_part_info = nil

		data_x.each do |part_key, part|
			inf_gain = @informationGain.information_gain(part, data_y)
			if inf_gain >= information_gain
				information_gain = inf_gain
				selected_key = part_key
			end
		end
		return [selected_key, information_gain]
	end

	def count_elem(vals)
		res = {}
		uniq_vals = vals.uniq
		uniq_vals.each do |val|
			res[val] = vals.count(val)
		end
		return res
	end

	def leaves_creation(part_feature, data_x, data_y, selected_key)
		tree = {}

		part_feature.uniq.each do |val|
			indexes = find_indexes(part_feature, val)
			sub_set = hash_select_indexes(data_x, indexes)
			vals = data_y.select.each_with_index { |val, i| indexes.include?(i) }

			tree[val] = { prediction: count_elem(vals).max_by {|k, v| v }[0] }
			
			if @informationGain.calc_part_loss(indexes, val, data_y) == 0
				indexes.each_with_index do |i, sub_ind|
					part_feature.delete_at(i - sub_ind)
					data_y.delete_at(i - sub_ind)
					data_x = hash_remove_index(data_x,  i - sub_ind)
				end
			else
				tree[val].merge(sub_set)
			end
		end
		return tree
	end

	def subset_loss_partition(data, data_y, part_feature, tree: {}, selected_key: nil)
		part_feature.uniq.each do |val|
			indexes = find_indexes(part_feature, val)
			sub_set = []
			part_data_y = []
			indexes.each { |i| part_data_y.push(data_y[i]) }
			sub_set = hash_select_indexes(data, indexes)
			tree[selected_key][val] = self.generate_tree(sub_set, part_data_y).merge({prediction: tree[selected_key][val][:prediction]}) if !data.empty?
		end
		return tree
	end

	def generate_tree(data, data_y, tree: {})
		return tree if data.empty?
		data_x = data.clone
		
		selected_key, information_gain = select_most_information_gain_feature(data_x, data_y)

		part_feature = data_x.delete(selected_key)

		leaves = leaves_creation(part_feature, data_x, data_y, selected_key)
		tree = subset_loss_partition(data_x, data_y, part_feature, tree: {selected_key => tree.merge(leaves)}, selected_key: selected_key)
		return tree
	end

	def self.classify(data, tree, parent_nod = nil)
		leaf = nil
		
		tree.each do |key, vals|
			return vals if vals.class != Hash
			lower_nod = vals[data[key]]
			if lower_nod.nil?
				parent_key, parent_vals = parent_nod.keys.first, parent_nod.values.first
				return parent_vals[data[parent_key]][:prediction] 
			elsif key == :prediction 
				return vals
			end
			leaf = lower_nod
			leaf = self.classify(data, lower_nod, {key => vals})
		end
					
		return leaf
	end

end

class Gini

	def calc_loss(data)
		uniq_vals = data.uniq.class == Array ? data.uniq : [data.uniq]
		loss = uniq_vals.sum do |val|
			prob = relative_frequence(data, val)
			prob**2
		end
		return 1 - loss
	end

	def information_gain(part, data)
		inf_gain = self.calc_loss(data) - self.calc_rem(part, data)
		return inf_gain
	end

	def calc_rem(part, data)
		uniq_vals = part.uniq.class == Array ? part.uniq : [part.uniq]

		rem = uniq_vals.sum do |val|
			part_size = part.size
			indexes = find_indexes(part, val)
			prob = indexes.size / part_size.to_f
			prob * self.calc_part_loss(indexes, val, data)
		end
		return rem
	end

	def calc_part_loss(indexes, val, data)
		part_data = indexes.map {|i| data[i] }
		part_loss = self.calc_loss(part_data)
		return part_loss
	end
end

class Entropy

	def calc_loss(data)
		uniq_vals = data.uniq.class == Array ? data.uniq : [data.uniq]
		loss = uniq_vals.sum do |val|
			prob = relative_frequence(data, val)
			-prob * (Math.log(prob) / Math.log(2))
		end
		return loss
	end

	def information_gain(part, data)
		inf_gain = self.calc_loss(data) - self.calc_rem(part, data)
		return inf_gain
	end

	def calc_rem(part, data)
		uniq_vals = part.uniq.class == Array ? part.uniq : [part.uniq]

		rem = uniq_vals.sum do |val|
			part_size = part.size
			indexes = find_indexes(part, val)
			prob = indexes.size / part_size.to_f
			prob * self.calc_part_loss(indexes, val, data)
		end
		return rem
	end

	def calc_part_loss(indexes, val, data)
		part_data = indexes.map {|i| data[i] }
		part_loss = self.calc_loss(part_data)
		return part_loss
	end

end