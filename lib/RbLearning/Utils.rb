module Utils

	def is_numeric?(val)
		val.to_i.to_s == val || val.to_f.to_s == val || val.is_a?(Integer) || val.is_a?(Float)
	end

	def find_indexes(arr, occ)
		match_occ = []
		arr.each_with_index { |val, i| match_occ << i if val == occ }
		return match_occ
	end

	EPSYLON = 0.000000001

end
