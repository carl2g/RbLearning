require './src/Statistics'
require './src/Utils'

class LeastS

	include Statistics
	include Utils

	attr_accessor :hashed_data, :size_x, :size_y

	def initialize(csv_file)
		content 		= CSV.read(csv_file)
		@hashed_data 	= {}
		content.shift.each { |key| @hashed_data[key] = [] }
		@hashed_data.each_with_index do |h, i|
			key = h.first
			@hashed_data[key] += content.map do |line|
				line[i]
			end
		end
		@size_x = @hashed_data.size
		@size_y = @hashed_data.first.last.size
	end

	def matrix(with = self.labels)
		newH = @hashed_data.select { |key, val| with.include?(key) }
		size_y = @size_y
		size_x = newH.keys.size
		arr = Matrix.new(size_y, size_x)
		(0...size_y).each do |x|
			newH.each_with_index do |h, y|
				key, val = h
				arr[x][y] = val[x].to_f
			end
		end
		return arr
	end

	def [](key)
		@hashed_data[key]
	end

	def []=(key, values)
		@hashed_data[key] = values
	end

	def describe_labels(labels = self.labels)
		labels.each do |label|
			describe_label(label)
		end
	end

	def describe_label(label)
		sorted_datas = self[label].map {|val| val.to_f }
		sorted_datas = sorted_datas.sort
		puts "=" * 45
		puts "\tName:\t\t#{label}"
		puts "\tCount:\t\t#{sorted_datas.select { |val| val if val != 0 }.size }"
		puts "\tMin:\t\t#{sorted_datas.first}"
		puts "\tMean:\t\t#{self.mean(sorted_datas)}"
		puts "\t25%:\t\t#{self.quarlite(sorted_datas, 1)}"
		puts "\t50%:\t\t#{self.quarlite(sorted_datas, 2)}"
		puts "\t75%:\t\t#{self.quarlite(sorted_datas, 3)}"
		puts "\tMax:\t\t#{sorted_datas.last}"
		puts "\tStandard dev:\t#{std_dev(sorted_datas)}"
		puts "\tSkewness:\t#{skewness(sorted_datas)}"
		puts "=" * 45
	end

	def addDumies(null_val = [])
		labels = []
		@hashed_data.each do |key, line|
			labels += line.each_with_index.map do |val, i|
				key if !self.is_numeric?(val)
			end
		end
		labels = labels.uniq.reject { |e| e.nil? }
		labels.each { |label| self.addDumie(label, null_val) }
	end

	def remove(label)
		@hashed_data.delete(label)
	end

	def addDumie(label, null_val = [])
		values = self.remove(label)
		exising_values = values.uniq.reject { |val| self.is_numeric?(val) || null_val.include?(val) }
		exising_values.each { |lab| @hashed_data[label + '_' + lab.to_s] = [] if !@hashed_data[lab] }
		(0...@size_y).each do |i|
			exising_values.each do |val|
				 @hashed_data[label + '_' + val.to_s] << (values[i] == val ? 1 : 0)
			end
		end
	end

	def labels
		@hashed_data.keys
	end
end
