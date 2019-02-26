// brent_test.go

package gopula

import (
	"math"
	"testing"
)

func parabol(x float64, a interface{}) float64 {
	return 1. + (x-a.(float64))*(x-a.(float64))
}

func fun0(x float64, k interface{}) float64 {
	return -math.Pow(x, k.(float64)) * math.Exp(-x)
}

func TestParabol(t *testing.T) {
	min := 2.0
	a := -10.
	b := 50.
	tol := 1e-8
	xmin, _, _ := BrentMinimizer(parabol, min, a, b, tol)
	if (xmin - min) > tol {
		t.Errorf("Minimum not found with given tolerance (expected %f, got %f)", a, xmin)
	}
}

func TestFun0(t *testing.T) {
	k := 7.0
	a := -10.
	b := 200.
	tol := 1e-2

	xmin, _, _ := BrentMinimizer(fun0, k, a, b, tol)
	if (xmin - k) > tol {
		t.Errorf("Minimum not found with given tolerance (expected %f, got %f)", k, xmin)
	}
}

func TestRoot(t *testing.T) {
	k := 7.0
	a := -10.
	b := 200.
	tol := 1e-8

	root, err := BrentRootFinder(fun0, k, a, b, tol)
	if err != nil {
		t.Error(err.Error())
	} else if (root - k) > tol {
		t.Errorf("Minimum not found with given tolerance (expected %f, got %f)", k, root)
	}
}
