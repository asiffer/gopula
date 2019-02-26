package gopula

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestLoadCsv(t *testing.T) {
	M, err := LoadCSV(csv, ',', false)
	n, p := M.Dims()
	if err != nil {
		t.Error(err.Error())
	} else if n != 2000 || p != 3 {
		t.Errorf("Bad matrix dimensions (expected (2000,3), got (%d, %d))", n, p)
	}
}

func TestSaveCsv(t *testing.T) {
	n := 4
	p := 4
	M := mat.NewDense(n, p, nil)
	M.Set(0, 1, 3.5)
	SaveCSV(M, "/tmp/test.csv", ',')
	N, err := LoadCSV("/tmp/test.csv", ',', false)
	if err != nil {
		t.Errorf("Error in loading previously saved matrix (%s)", err.Error())
	}
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			if N.At(i, j) != M.At(i, j) {
				t.Errorf("Bad matrix element [%d, %d]: expected %f, got %f", i, j, M.At(i, j), N.At(i, j))
			}
		}
	}
}
