module Utils

	def is_numeric?(val)
		val.to_i.to_s == val || val.to_f.to_s == val || val.is_a?(Integer) || val.is_a?(Float)
	end

	EPSYLON = 0.000000001

end
