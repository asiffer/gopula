package gopula

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

var (
	testCSV = "resources/data.csv"
)

func TestSaveCsv(t *testing.T) {
	n := 4
	p := 3
	M := mat.NewDense(n, p, nil)
	M.Set(0, 1, 3.5)
	err := SaveCSV(M, testCSV, ',')
	if err != nil {
		t.Errorf("Error in saving matrix (%s)", err.Error())
	}
}

func TestLoadCsv(t *testing.T) {
	M, err := LoadCSV(testCSV, ',', false)
	n, p := M.Dims()
	if err != nil {
		t.Error(err.Error())
	} else if n != 4 || p != 3 {
		t.Errorf("Bad matrix dimensions (expected (4,3), got (%d, %d))", n, p)
	}
}
