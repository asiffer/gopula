// utils_test.go

package gopula

import (
	"math"
	"testing"
)

var (
	csv = "/home/asr/Documents/Work/go/src/gopula/clayton.csv"
)

func TestProd(t *testing.T) {
	vector := []float64{1, 2, 3, 4, 5, 6}
	p := prod(vector)
	if p != 720. {
		t.Errorf("bad product computation, expected 720, got %f", p)
	}
}

func TestSum(t *testing.T) {
	vector := []float64{1, 2, 3, 4, 5, 6}
	s := sum(vector)
	if s != 21. {
		t.Errorf("bad sum computation, expected 21, got %f", s)
	}
}

func TestMax(t *testing.T) {
	vector := []float64{-7., 3.5, 0., 1. / 3.}
	m := max(vector)
	if m != 3.5 {
		t.Errorf("bad max computation, expected 3.5, got %f", m)
	}
}

func TestMin(t *testing.T) {
	vector := []float64{-7., 3.5, 0., 1. / 3.}
	m := min(vector)
	if m != -7 {
		t.Errorf("bad min computation, expected -7., got %f", m)
	}
}

func TestArange(t *testing.T) {
	r := 0.
	step := 0.1
	vector := arange(r, r+2., step)
	for _, x := range vector {
		if x != r {
			t.Errorf("bad arange creation, expected %f, got %f", r, x)
		}
		r += step
	}
}

func TestOnes(t *testing.T) {
	vector := []float64{0, 1, 2, 3, 4, 5, 6}
	square := []float64{0, 1, 4, 9, 16, 25, 36}
	for i, x := range pow(vector, 2) {
		if x != square[i] {
			t.Errorf("Bad values at index %d: expected %f, got %f", i, square[i], x)
		}
	}
}

func TestScalarAdd(t *testing.T) {
	v := ones(50)
	v = scalarAdd(v, -1.0)
	for i, x := range v {
		if x != 0.0 {
			t.Errorf("Bad values at index %d: expected %f, got %f", i, 0.0, v[i])
		}
	}
}

func TestPow(t *testing.T) {
	v := ones(50)
	v = scalarAdd(v, -1.0)
	for i, x := range v {
		if x != 0.0 {
			t.Errorf("Bad values at index %d: expected %f, got %f", i, 0.0, v[i])
		}
	}
}

func TestScalarDiv(t *testing.T) {
	vector := []float64{2., 4., 6., 8., 10.}
	d := scalarDiv(vector, 2)
	for i := 0; i < len(vector); i++ {
		if d[i] != vector[i]/2. {
			t.Errorf("bad division computation, expected %.1f, got %.1f", vector[i]/2., d[i])
		}
	}
}

func TestJoin(t *testing.T) {
	vector := []float64{0.5, 1, 2}
	if join(vector, ",") != "0.5000000000,1.0000000000,2.0000000000" {
		t.Errorf("Bad join: expected '0.500000,1.000000,2.000000', got %s", join(vector, ","))
	}
}

func TestLog(t *testing.T) {
	vector := []float64{0.5, 1, 2}
	p := log(vector)
	if p[0] != math.Log(0.5) || p[1] != 0 || p[2] != math.Log(2) {
		t.Errorf("bad log computation, expected [-0.693, 0, 0.693], got %f", p)
	}
}

func TestFactorial(t *testing.T) {
	if factorial(6) != 720. {
		t.Errorf("bad factorial computation, expected 720, got %d", factorial(6))
	}
}

func TestEuclid(t *testing.T) {
	a := 103
	b := 17
	q, r := euclid(a, b)
	if q != 6 && r != 1 {
		t.Errorf("Bad euclid division: expected (q,r) = (6,1), got (%d,%d)", q, r)
	}
}
