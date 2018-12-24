module Utils

	def is_numeric?(val)
		val.to_i.to_s == val || val.to_f.to_s == val || val.class == (Numeric) || val.class == (Float)
	end

end
