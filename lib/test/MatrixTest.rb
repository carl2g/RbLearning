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

    def test_sub_1
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
        res = (m1 - m2)
        ass = (res.matrix == [0, 0, 0, 0, 0, 0, 0, 0])
        assert(ass, "Test sub_1 failed")
    end

    def test_sub_2
        m1 = Matrix.set([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])
        m2 = Matrix.set([
            [2, 1, 4, 3],
            [6, 5, 7, 7]
        ])
        res = (m1 - m2)
        ass = (res.matrix == [-1, 1, -1, 1, -1, 1, 0, 1])
        assert(ass, "Test sub_2 failed")
    end

    def test_add_1
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
        res = (m1 + m2)
        ass = (res.matrix == [2, 4, 6, 8, 10, 12, 14, 16])
        assert(ass, "Test add_1 failed")
    end

    def test_add_2
        m1 = Matrix.set([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])
        m2 = Matrix.set([
            [2, 1, 4, 3],
            [6, 5, 7, 7]
        ])
        res = (m1 + m2)
        ass = (res.matrix == [3, 3, 7, 7, 11, 11, 14, 15])
        assert(ass, "Test add_2 failed")
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
