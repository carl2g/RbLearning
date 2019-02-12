require 'csv'
require 'bitmap'

class Knn

	def predict(data, m, c_exp_res, nb = 100)
		res = {}
		m.matrix.each_with_index do |line, i|
			diff = 0
			line.each_with_index do |val, x|
				diff += Math.sqrt((data[x] - val.to_f)**2)
			end
			res[i] = { lab: c_exp_res[i], diff: diff }
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

end
