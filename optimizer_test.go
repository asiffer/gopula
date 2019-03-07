// optimizer_test.go

package gopula

import (
	"fmt"
	"math"
	"testing"
)

func TestInitOptimizer(t *testing.T) {
	title("Optimizers")
}

func parabol(x float64, a interface{}) float64 {
	return 1. + (x-a.(float64))*(x-a.(float64))
}

func fun0(x float64, k interface{}) float64 {
	return -math.Pow(x, k.(float64)) * math.Exp(-x)
}

func funTestRoot(x float64, k interface{}) float64 {
	kf := k.(float64)
	return 1 - kf*math.Pow(x, kf)
}

func TestParabol(t *testing.T) {
	min := 2.0
	a := -10.
	b := 50.
	tol := 1e-8
	xmin, _, _, err := BrentMinimizer(parabol, min, a, b, tol)
	if err != nil {
		t.Fatal(err)
	}
	if (xmin - min) > tol {
		t.Errorf("Minimum not found with given tolerance (expected %f, got %f)", a, xmin)
	}
}

func TestFun0(t *testing.T) {
	k := 7.0
	a := -10.
	b := 200.
	tol := 1e-2

	xmin, _, _, err := BrentMinimizer(fun0, k, a, b, tol)
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

	root, err := BrentRootFinder(funTestRoot, k, a, b, tol)
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

	root, err := Bisection(funTestRoot, k, a, b, tol)
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

	root, err := Secant(funTestRoot, k, a, b, tol)
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
	arch := NewCopula("Clayton", 5.)
	M, err := LoadCSV(claytonSample, ',', false)
	if err != nil {
		t.Fatal(err)
	}
	a, b := arch.copula.ThetaBounds()

	fmt.Printf("\nMethod %-12s %-12s %-12s\n", "𝜃*", "ℓ", "fEval")
	fmt.Println("--------------------------------------")
	thetaBest, llhood, nit, _ := BFGS(arch.logLikelihoodToMinimize, M, 2.5)
	fmt.Printf("BFGS   %-12.6f %-12.6f %-12d\n", thetaBest, -llhood, nit)
	thetaBest, llhood, nit, _ = BrentMinimizer(arch.logLikelihoodToMinimize, M, a, b, 1e-6)
	fmt.Printf("Brent  %-12.6f %-12.6f %-12d\n", thetaBest, -llhood, nit)
	fmt.Println("--------------------------------------")
}
