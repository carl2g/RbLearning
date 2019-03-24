require 'test/unit'
require './lib/RbLearning/Matrix'

class MatrixTest < Test::Unit::TestCase

    def test_mult_1
    	m1 = Matrix.set([
    		[1, 2],
    		[3, 4],
    		[5, 6],
    		[7, 8]
    	])
    	m2 = Matrix.set([
    		[1, 2],
    		[3, 4],
    		[5, 6],
    		[7, 8]
    	])
    	res = (m1 ** m2)
    	ass = (res.matrix == [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0])
    	assert(ass, "Test mult_1 failed")
  	end

    def test_mult_2
        m1 = Matrix.set([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])
        m2 = Matrix.set([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ])
        res = (m1 ** m2)
        ass = (res.matrix == [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0])
        assert(ass, "Test mult_2 failed")
    end

    def test_dot_1
        m1 = Matrix.set([
            [1, 2],
            [3, 4]
        ])
        m2 = Matrix.set([
            [1, 2],
            [3, 4]
        ])
        res = (m1 * m2)
        ass = (res.matrix == [7, 10, 15, 22])
        assert(ass, "Test dot_1 failed")
    end

    def test_dot_2
        m1 = Matrix.set([
            [1, 2, 3],
            [4, 5, 6]
        ])
        m2 = Matrix.set([
            [1, 2],
            [3, 4],
            [5, 6],
        ])
        res = (m1 * m2)
        ass = (res.matrix == [22, 28, 49, 64])
        assert(ass, "Test dot_2 failed")
    end

    def test_dot_3
        m1 = Matrix.set([
            [1, 2, 3],
            [4, 5, 6]
        ])
        m2 = Matrix.set([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        res = (m1 * m2)
        ass = (res.matrix == [30, 36, 42, 66, 81, 96])
        assert(ass, "Test dot_3 failed")
    end

end
