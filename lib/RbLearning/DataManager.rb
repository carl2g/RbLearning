require_relative './Statistic'
require_relative './Utils'
require 'csv'

class DataManager

	include Statistic
	include Utils

	attr_accessor :data, :size_x, :size_y

	# Create DataManager Object
	#
	# == Parameters:
	# Take a csv file
	#
	# == Returns:
	# New DataManager Object
	#
	def initialize(csv_file)
		content = CSV.read(csv_file)
		@data 	= {}
		content.shift.each { |key| @data[key] = [] }
		@data.each_with_index do |h, i|
			key = h.first
			@data[key] += content.map do |line|
				line[i]
			end
		end
		@size_x = @data.size
		@size_y = @data.first.last.size
	end

	def standerdize
		self.labels.each do |l|
			self[l].map!(&:to_i)
			self[l] = Statistic::standerdize(self[l])
		end
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
		newH = @data.select { |key, val| with.include?(key) }
		size_y = @size_y
		size_x = newH.keys.size
		m = Matrix.new(size_y, size_x)
		
		(0...size_x).each do |x|
			arr = newH[newH.keys[x]]
			arr.each_with_index do |val, y|
				m[y, x] = val.to_f
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
		return matrixGenerate
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
		@data[key]
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
		@data[key] = values
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
		@data.each do |key, line|
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
		@data.delete(label)
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
		exising_values.each do |lab| 
			new_lab = label + '_' + lab.to_s
			new_feature = {new_lab => []}
			self.data = new_feature.merge(self.data)
		end
		(0...@size_y).each do |i|
			exising_values.each do |val|
				 self.data[label + '_' + val.to_s] << (values[i] == val ? 1 : 0)
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
		@data.keys
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
		offset = 0
		rm_rows = []
		self[key].each_with_index do |val, i|
			if val == rm_val
				rm_rows << removeAt(i - offset) 
				offset += 1
			end
		end
		return rm_rows
	end

	# Remove rows at a given index
	#
	# == Parameters:
	# index to delete row
	#
	def removeAt(i)
		rm_data = []
		self.labels.each do |l|
			rm_data << self[l].delete_at(i)
		end
		updateSize
		return rm_data
	end

	def getAt(i)
		data = []
		self.labels.each do |l|
			data << self[l][i]
		end
		return data
	end

	def replace_val_label(lab, val_to_replace, new_val)
		self[lab] = self[lab].map.each_with_index do |val, i|
				val == val_to_replace ? new_val : val
		end
		return self[lab]
	end

	def replace_val(val_to_replace, new_val)
		self.data.each do |key, val|
			val.each_with_index.each do |v, i|
				self[key][i] = new_val if v == val_to_replace
			end
		end
	end

	def printData
		self.labels.each do |l|
			if self.labels.last == l
				print "#{l}"
			else
				print "#{l}, "
			end
		end
		puts
		(0...self.size_y).each do |r|
			self.getAt(r).each do |v|
				print "#{v}, "
			end
			puts
		end
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
		self.remove(key) if self[key].count(val) >= (@size_y / 100.0) * (perc * 100)
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
	def self.batch(y: [], x: matrix, batch_size: 24)
		Random.srand
		indexes = (0...batch_size).map { Random.rand(0...x.size_y) }
		batch_x = Matrix.set(indexes.map do |i|
			x[i]
		end)
		batch_y = Matrix.set(indexes.map do |i|
			y[i]
		end)
		return [batch_x, batch_y]
	end

	def split(arr, target, random: false)
		data = []
		original_size = self.size_y
		v = 0
		added_inds = []
		offset = 0
		arr.each_with_index do |perc, it|
			x = []
			y = []
			nb = (original_size * perc).round
			(0...nb).each do |i|
				ind = random ? Random.rand(0...original_size) : i + offset
				if added_inds.include?(ind)
					redo
				else 
					x << self.getAt(ind)
					y << target[ind]
					added_inds << ind
				end
			end
			offset += nb
			data[v] = Matrix.set(x)
			data[v + 1] = Matrix.set(y)
			v = v + 2
		end
		return data
	end

	private

		def updateSize
			@size_x = @data.size
			@size_y = @data.first.last.size
		end

end
