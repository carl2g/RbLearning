require_relative './Matrix'
require_relative './Gauss.rb'

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
	squared_res = mat.transpose  * res
	newM = mat.transpose * mat
	newM = newM << squared_res
	gauss = GaussElemination(newM)
	return triangleEquationSolver(gauss)
end

def getPrediction(mat, coefs)
	prediction = [0] * mat.size_y
	(0...mat.size_y).each do |y|
		(0...mat.size_x).each do |x|
			prediction[y] += mat[y, x] * coefs[x]
		end
	end
	return prediction
end

def leastSquaredError(mat, coef, c_expected_res)
	err = 0.0

	(0...mat.size_y).each do |y|
		found_price = 0.0
		(0...mat.size_x).each do |x|
			found_price += mat[y, x] * coef[x]
		end
		err += (c_expected_res[y, 0] - found_price)**2
	end
	err = err / c_expected_res.size_y
	return Math.sqrt(err)
end
