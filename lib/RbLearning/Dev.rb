def triangleEquationSolver(mat, feature_size = mat.size_y)
	coefs = Matrix.new(feature_size, 1, 0)

	(feature_size - 1).downto(0) do |y|
		res = mat[y].last
		(feature_size).downto(y) do |x|
			res -= mat[y, x] * coefs[y, 0]
		end
		if coefs[y, 0] && mat[y, y] != 0
			coefs[y, 0] = res / mat[y, y]
		else
			coefs[y, 0] = 0.0
		end
	end
	return coefs
end

def poylnomialRegression(mat, res)
	res = mat.transpose * res
	m = mat.transpose * mat
	m = m << res
	gauss = GaussElemination(m)
	return triangleEquationSolver(gauss)
end

def switchLine(mat, y, x)
	max = mat[y, x]
	tmp = y
	(y...mat.size_y).each do |l|
		if mat[l, x].abs > max
			tmp = l
			max = mat[l, x]
		end
	end
	save = mat[tmp]
	mat.setLine(tmp, mat[y])
	mat.setLine(y, save)
end

def findPivot(mat, y, x)
	switchLine(mat, y, x)
	return [y, x]
end

def GaussElemination(mat, size_y = mat.size_y, size_x = mat.size_x)
	newMat = Matrix.setVectorizedMatrix(mat.matrix, mat.size_y, mat.size_x)

	y = 0
	x = 0

	while y < (size_y - 1)
		y, x = findPivot(newMat, y, y)
		pivot = newMat[y, x]
		if pivot != 0
			((y + 1)...size_y).each do |d|
				var 	= newMat[d, x]
				mult 	= var.to_f / pivot.to_f
				i 	= x
				while i < size_x
					newMat[d, i] = (newMat[d, i] - newMat[y, i] * mult).to_f
					i = i + 1
				end
			end
		end
		y = y + 1
	end
	return newMat
end