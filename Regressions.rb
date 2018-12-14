require './Matrix'
require './Gauss.rb'

def triangleEquationSolver(mat, vect_size = mat.size_x - 1)
	vect_x = Array.new(vect_size) { |i| 0.0 }

	(mat.size_y - 1).downto(0) do |y|
		res = mat[y].last
		(vect_size - 1).downto(y) do |x|
			res -= mat[y][x] * vect_x[x]
		end
		if vect_x[y] && mat[y][y] != 0
			vect_x[y] = res / mat[y][y]
		else
			vect_x[y] = 0.0
		end
	end
	return vect_x
end

def poylnomialRegression(mat, res)
	squared_res = res * mat.transpose
	newM = mat * mat.transpose
	newM = newM + squared_res
	gauss = GaussElemination(newM)
	return triangleEquationSolver(gauss)
end
