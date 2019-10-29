module Utils

	def is_numeric?(val)
		val.to_i.to_s == val || val.to_f.to_s == val || val.is_a?(Integer) || val.is_a?(Float)
	end

	def find_indexes(arr, occ)
		match_occ = []
		arr.each_with_index { |val, i| match_occ << i if val == occ }
		return match_occ
	end

	def hash_remove_index(h, ind)
		h.each do |key, arr|
			arr.delete_at(ind)
		end
		return h
	end

	def hash_select_indexes(h, indexes)
		new_h = {}
		h.each do |key, arr|
			indexes.each do |ind|
				if new_h[key] 
					new_h[key].push(arr[ind])
				else
					new_h[key] = [arr[ind]]
				end
			end
		end
		return new_h
	end

	def hash_select_index(h, ind)
		new_h = {}
		h.each do |key, arr|
			new_h[key] = arr[ind]
		end
		return new_h
	end

	EPSYLON = 0.000000001

end
