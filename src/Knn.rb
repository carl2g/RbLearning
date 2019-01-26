require 'csv'
require 'bitmap'

class Knn

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

	def predict(data, m, exp_res, nb = 100)
		res = {}
		m.matrix.each_with_index do |line, i|
			diff = 0
			line.each_with_index do |val, x|
				diff += Math.sqrt((data[x] - val.to_f)**2)
			end
			res[i] = { lab: exp_res[i], diff: diff }
		end
		res = res.sort_by {|i, res|  res[:diff] }
		final_res = {}
		res.first(nb).each do |ind, h|
			final_res[h[:lab]] = 0 if final_res[h[:lab]].nil?
			final_res[h[:lab]] += 1
		end
		final_res = final_res.sort_by {|key, val| val }
		return final_res.last
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

	def remove(label)
		@hashed_data.delete(label)
	end

	def [](key)
		@hashed_data[key]
	end

	def []=(key, values)
		@hashed_data[key] = values
	end

	def labels
		@hashed_data.keys
	end

end
