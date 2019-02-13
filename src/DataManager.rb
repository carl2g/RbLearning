require_relative './Statistics'
require_relative './Utils'

class DataManager

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
		m = Matrix.new(size_y, size_x)
		(0...size_y).each do |x|
			newH.each_with_index do |h, y|
				key, val = h
				m[x, y] = val[x].to_f
			end
		end
		return m
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

	def normalize
		self.labels.each do |key|
			self[key].map! { |e| e.to_f }
			max = self[key].max
			min = self[key].min
			norm = max < min.abs ? min : max
			self[key].map! { |e| norm != 0 ? e / norm : e }
		end
	end

	def removeRaws(rm)
		self.labels.each do |l|
			removeRaw(l, rm)
		end
	end

	def removeRaw(key, rm)
		self[key].each_with_index.reverse_each do |val, i|
			removeAt(i) if val == rm
		end
	end

	def removeAt(i)
		self.labels.each do |l|
			self[l].delete_at(i)
		end
		updateSizeInfo
	end

	def removeKeysNullVall(perc, val = nil, keys = self.labels)
		keys.each do |key|
			removeKeyNullVal(key, perc, val)
		end
	end

	def removeKeyNullVal(key, perc, val = nil)
		self.remove(key) if self[key].count(val) >= (@size_y / 100.0) * perc
	end

	private

		def updateSizeInfo
			@size_x = @hashed_data.size
			@size_y = @hashed_data.first.last.size
		end

end
