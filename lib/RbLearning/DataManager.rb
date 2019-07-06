require_relative './Statistics'
require_relative './Utils'
require 'csv'

class DataManager

	include Statistics
	include Utils

	attr_accessor :hashed_data, :size_x, :size_y, :mat

	# Create DataManager Object
	#
	# == Parameters:
	# Take a csv file
	#
	# == Returns:
	# New DataManager Object
	#
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

	# Generate a Matrix Object using DataManager Labels
	#
	# == Parameters:
	# Take array of label witch will be used to generate a Matrix,
	# by default all labels of DataManager Object is selected
	#
	# == Returns:
	# Matrix Object
	#
	def matrixGenerate(with = self.labels)
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

	# Return associated Matrix to DataManager Object
	# else call self.matrixGenerate to generate one
	# == Returns:
	# Matrix Object
	#
	def matrix
		@mat ||= self.matrixGenerate
		return @mat
	end

	# Get all values for a given label
	#
	# == Parameters:
	# Take a key (label)
	#
	# == Returns:
	# Array of value
	#
	def [](key)
		@hashed_data[key]
	end

	# Set values for a given label
	#
	# == Parameters:
	# Take a key (label) and a array of value
	#
	# == Returns:
	# New Array of value
	#
	def []=(key, values)
		@hashed_data[key] = values
	end

	# Replace lables with non numeric values by dumies variables
	#
	# == Parameters:
	# Take a key (label)
	#
	# == Returns:
	# New labels
	#
	def addDumies
		labels = []
		@hashed_data.each do |key, line|
			labels += line.each_with_index.map do |val, i|
				key if !self.is_numeric?(val)
			end
		end
		labels = labels.uniq.reject { |e| e.nil? }
		labels.each { |label| self.addDumie(label) }
		return self.labels
	end

	def keepLabels(labels)
		my_labs = self.labels
		rm_labs = my_labs - labels
		rm_labs.each do |l|
			remove(l)
		end
	end

	# Remove label
	#
	# == Parameters:
	# Take a key (label)
	#
	# == Returns:
	# Removed data
	#
	def remove(label)
		@hashed_data.delete(label)
	end

	# Replace a given lable with non numeric values by dumies variables
	#
	# == Parameters:
	# Take a key (label)
	#
	# == Returns:
	# New labels
	#
	def addDumie(label)
		values = self.remove(label)
		exising_values = values.uniq
		exising_values.each { |lab| @hashed_data[label + '_' + lab.to_s] = [] if !@hashed_data[lab] }
		(0...@size_y).each do |i|
			exising_values.each do |val|
				 @hashed_data[label + '_' + val.to_s] << (values[i] == val ? 1 : 0)
			end
		end
		return self.labels
	end

	# Get lables from dataManager
	#
	# == Returns:
	# labels
	#
	def labels
		@hashed_data.keys
	end

	# Normalize values, divide all values by the max value for a given label
	#
	def normalize
		self.labels.each do |key|
			self[key].map! { |e| e.to_f }
			max = self[key].max.abs
			min = self[key].min.abs
			norm = max < min ? min : max
			self[key].map! { |e| norm != 0 ? e / norm : e }
		end
		return nil
	end

	# Remove rows having given value
	#
	# == Parameters:
	# value to delete row
	#
	def removeRows(rm_val)
		self.labels.each do |l|
			removeRow(l, rm_val)
		end
		return nil
	end

	# Remove rows having given value for a given label
	#
	# == Parameters:
	# key (label)
	# value to delete row
	#
	def removeRow(key, rm_val)
		self[key].each_with_index.reverse_each do |val, i|
			removeAt(i) if val == rm_val
		end
		return nil
	end

	# Remove rows at a given index
	#
	# == Parameters:
	# index to delete row
	#
	def removeAt(i)
		self.labels.each do |l|
			self[l].delete_at(i)
		end
		updateSizeInfo
		return nil
	end

	# Remove labels having a given percent of given value for given labels
	#
	# == Parameters:
	# percentage of given value
	# value to apply function default: nil
	# lebels to apply function default: self.labels
	#
	# == Returns:
	# labels
	#
	def removeKeysNullVall(perc, val = nil, keys = self.labels)
		keys.each do |key|
			removeKeyNullVal(key, perc, val)
		end
		return self.labels
	end

	# Remove label having a given percent of given value for given labels
	#
	# == Parameters:
	# percentage of given value
	# value to apply function default: nil
	# lebels to apply function default: self.labels
	#
	# == Returns:
	# labels
	#
	def removeKeyNullVal(key, perc, val = nil)
		self.remove(key) if self[key].count(val) >= (@size_y / 100.0) * perc
	end

	# Generate batch of given size from matrix used to train NeuralNet
	#
	# == Parameters:
	# Expected result (for training)
	# Size of batch default: 24
	#
	# == Returns:
	# 2 Matrix object representing the parameter x and the result y
	#
	def batch(data_y, data_x: matrix, batch_size: 24)
		Random.srand
		indexes = (0...batch_size).map { Random.rand(0...data_x.size_y) }
		batch_x = Matrix.set(indexes.map do |i|
			data_x[i]
		end)
		batch_y = Matrix.set(indexes.map do |i|
			data_y[i]
		end)
		return [batch_x, batch_y]
	end

	private

		def updateSizeInfo
			@size_x = @hashed_data.size
			@size_y = @hashed_data.first.last.size
		end

end
