// optimizer_test.go

package gopula

import (
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestInitOptimizer(t *testing.T) {
	title("Optimizers")
}

func parabol(x float64, M *mat.Dense) float64 {
	a := M.At(0, 0)
	return 1. + (x-a)*(x-a)
}

func fun0(x float64, M *mat.Dense) float64 {
	k := M.At(0, 0)
	return -math.Pow(x, k) * math.Exp(-x)
}

func funTestRoot(x float64, M *mat.Dense) float64 {
	k := M.At(0, 0)
	return 1 - k*math.Pow(x, k)
}

func asDense(f float64) *mat.Dense {
	return mat.NewDense(1, 1, []float64{f})
}

func TestParabol(t *testing.T) {
	min := asDense(2.0)
	a := -10.
	b := 50.
	tol := 1e-8
	xmin, _, _, err := BrentMinimizer(parabol, min, a, b, tol)
	if err != nil {
		t.Fatal(err)
	}
	if (xmin - min.At(0, 0)) > tol {
		t.Errorf("Minimum not found with given tolerance (expected %f, got %f)", a, xmin)
	}
}

func TestFun0(t *testing.T) {
	k := 7.0
	a := -10.
	b := 200.
	tol := 1e-2

	xmin, _, _, err := BrentMinimizer(fun0, asDense(k), a, b, tol)
	if err != nil {
		t.Fatal(err)
	}
	if (xmin - k) > tol {
		t.Errorf("Minimum not found with given tolerance (expected %f, got %f)", k, xmin)
	}
}

func TestBrentRootFinder(t *testing.T) {
	checkTitle("Testing Brent Root Finder...")
	k := 7.0
	a := -5.
	b := 10.
	tol := 1e-4

	sol := math.Pow(1/k, 1/k)

	root, err := BrentRootFinder(funTestRoot, asDense(k), a, b, tol)
	if err != nil {
		t.Log(err)
	}
	if math.Abs(root-sol) > tol {
		testERROR()
		t.Errorf("Root not found with given tolerance (expected %f, got %f)", sol, root)
	} else {
		testOK()
	}
}

func TestBisection(t *testing.T) {
	checkTitle("Testing Bisection...")
	k := 7.0
	a := -5.
	b := 10.
	tol := 1e-8

	sol := math.Pow(1/k, 1/k)

	root, err := Bisection(funTestRoot, asDense(k), a, b, tol)
	if err != nil {
		t.Log(err)
	}
	if math.Abs(root-sol) > tol {
		testERROR()
		t.Errorf("Root not found with given tolerance (expected %f, got %f)", sol, root)
	} else {
		testOK()
	}
}

func TestSecant(t *testing.T) {
	checkTitle("Testing Secant...")
	k := 3.0
	a := 0.1
	b := 10.
	tol := 1e-4

	sol := math.Pow(1/k, 1/k)

	root, err := Secant(funTestRoot, asDense(k), a, b, tol)
	if err != nil {
		t.Log(err)
	}
	if math.Abs(root-sol) > tol {
		testERROR()
		t.Errorf("Root not found with given tolerance (expected %f, got %f)", sol, root)
	} else {
		testOK()
	}
}

func TestOptimizerComparison(t *testing.T) {
	checkTitle("Comparison between Brent and BFGS...\n")
	arch, err := NewCopula("Clayton", 5.)
	if err != nil {
		t.Fatal(err)
	}
	data, err := claytonFS.ReadFile("tests/samples/clayton_2.csv")
	if err != nil {
		t.Fatal(err)
	}
	M, err := loadCSVFromBytes(data, ',')
	if err != nil {
		t.Fatal(err)
	}
	_, b := arch.copula.ThetaBounds()
	if math.IsInf(b, 1) {
		b = 20.0
	}

	fmt.Printf("\nMethod %-12s %-12s %-12s\n", "𝜃*", "ℓ", "fEval")
	fmt.Println("--------------------------------------")
	thetaBest, llhood, nit, _ := BFGS(arch.logLikelihoodToMinimize, M, 3.3)
	fmt.Printf("BFGS   %-12.6f %-12.6f %-12d\n", thetaBest, -llhood, nit)
	thetaBest, llhood, nit, _ = BrentMinimizer(arch.logLikelihoodToMinimize, M, 1, b, 1e-6)
	fmt.Printf("Brent  %-12.6f %-12.6f %-12d\n", thetaBest, -llhood, nit)
	fmt.Println("--------------------------------------")
}
