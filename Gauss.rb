require './Matrix'
require './Regressions'

def switchLine(mat, y, x)
	max = mat[y][x]
	tmp = y
	(y...mat.size_y).each do |l|
		if mat[l][x].abs > max
			tmp = l
			max = mat[l][x]
		end
	end
	save = mat[tmp]
	mat[tmp] = mat[y]
	mat[y] = save
end

def findPivot(mat, y, x)
	return [y, x] if mat[y] && mat[y][y] && mat[y][y] != 0
	while x < mat.size_x
		switchLine(mat, y, x)
		break if mat[y][x] != 0
		x = x + 1
	end
	return [y, x]
end

def GaussElemination(mat, size_y = mat.size_y, size_x = mat.size_x)
	newMat = Matrix.new()
	newMat.set(mat.matrix)

	y = 0
	x = 0

	while y < (size_y - 1)
		y, x = findPivot(newMat, y, y)
		pivot = newMat[y][x]
		if pivot != 0
			((y + 1)...size_y).each do |d|
				var 	= newMat[d][x]
				mult 	= var.to_f / pivot.to_f
				i 	= x
				while i < size_x
					newMat[d][i] = (newMat[d][i] - newMat[y][i] * mult).to_f
					i = i + 1
				end
			end
		end
		y = y + 1
	end
	return newMat
end
