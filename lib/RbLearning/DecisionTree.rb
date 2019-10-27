require_relative './ActivFunc'
require_relative './LossFunc'

class DecisionTree

	include Statistic
	attr_accessor :informationGain

	def initialize(informationGain: Entropy)
		@informationGain = informationGain
	end

	def generate_tree(data_x, data_y)
		f = data_y
		information_gain = 0
		selected_key = nil
		data_x.each do |key, arr|
			f_gain = informationGain.information_gain(arr, f)
			if f_gain >= information_gain
				information_gain = f_gain
				selected_key = key
			end
		end
		arr = data_x.delete(selected_key)
		puts "#{selected_key}: #{information_gain}"
		puts "=" * 40
		self.generate_tree(data_x, arr) if !data_x.empty?
	end

end

module Entropy

	def calc_entropy(data)
		uniq_vals = data.uniq.class == Array ? data.uniq : [data.uniq]
		entropy = uniq_vals.sum do |val|
			prob = relative_frequence(data, val)
			-prob * (Math.log(prob) / Math.log(2))
		end
		return entropy
	end

	def information_gain(part, data)
		inf_gain =  self.calc_entropy(data) - self.calc_part_entropy(part, data)
		puts "entropy of feature: #{self.calc_entropy(part)}"
		puts "partial entropy: #{self.calc_part_entropy(part, data)}"
		puts "Information gain: #{inf_gain}"

		return inf_gain
	end

	def calc_part_entropy(part, data)
		uniq_vals = part.uniq.class == Array ? part.uniq : [part.uniq]
		part_size = part.size
		inf_gain = uniq_vals.sum do |val|
			indexes = find_indexes(part, val)
			prob = indexes.size / part_size.to_f
			part_data = indexes.map {|i| data[i] }
			prob * self.calc_entropy(part_data)
		end
		return inf_gain
	end

end