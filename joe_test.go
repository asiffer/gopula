// joe_test.go

package gopula

import (
	"fmt"
	"math"
	"testing"
)

var (
	joeSample = "resources/joe_7.50.csv"
)

func TestInitJoe(t *testing.T) {
	title("Joe")
}

func TestJoeDistribution(t *testing.T) {
	AC := NewCopula("joe", 2.0)

	checkTitle("Checking pdf...")
	pdf := AC.Pdf([]float64{0.5, 0.5})
	if math.Abs(pdf-1.242) > 1e-3 {
		t.Errorf("Bad pdf computation, expected 1.242, got %f", pdf)
		testERROR()
	} else {
		testOK()
	}

	checkTitle("Checking cdf...")
	AC.theta = 2.5
	cdf := AC.Cdf([]float64{0.25, 0.75})
	if math.Abs(cdf-0.24023) > 1e-5 {
		t.Errorf("Bad cdf computation, expected 0.24023, got %f", cdf)
		testERROR()
	} else {
		testOK()
	}

	checkTitle("Checking logpdf...")
	AC.theta = 5.4
	lpdf := AC.LogPdf([]float64{0.35, 0.85})
	if math.Abs(lpdf+4.51813) > 1e-5 {
		t.Errorf("Bad logpdf computation, expected -4.51813, got %f", lpdf)
		testERROR()
	} else {
		testOK()
	}
}

func TestJoeRadialCdf(t *testing.T) {
	checkTitle("Checking radial cdf...")
	theta := 1.45
	AC := NewCopula("joe", theta)
	rcdf := AC.RadialCdf(0.5, 3)
	if math.Abs(rcdf-0.142949) > 1e-5 {
		t.Errorf("Bad radial cdf computation, expected cdf = 0.142949, got %f", rcdf)
		testERROR()
	} else {
		testOK()
	}
}

func TestJoeRadialPpf(t *testing.T) {
	theta := 2.10
	AC := NewCopula("joe", theta)
	rppf25 := AC.RadialPpf(0.25, 3)
	rppf50 := AC.RadialPpf(0.5, 3)
	rppf75 := AC.RadialPpf(0.75, 3)

	checkTitle("Checking radial ppf (25%)...")
	if math.Abs(AC.RadialCdf(rppf25, 3)-0.25) > 1.e-6 {
		t.Errorf("Bad radial 25%% ppf computation, expected quantile: 3.927744, got %f", rppf25)
		testERROR()
	} else {
		testOK()
	}
	checkTitle("Checking radial ppf (50%)...")
	if math.Abs(AC.RadialCdf(rppf50, 3)-0.50) > 1.e-6 {
		t.Errorf("Bad radial 50%% ppf computation, expected quantile: 12.784697, got %f", rppf50)
		testERROR()
	} else {
		testOK()
	}
	checkTitle("Checking radial ppf (75%)...")
	if math.Abs(AC.RadialCdf(rppf75, 3)-0.75) > 1.e-6 {
		t.Errorf("Bad radial 75%% ppf computation, expected quantile: 62.839318, got %f", rppf75)
		testERROR()
	} else {
		testOK()
	}
}

func TestJoeSampling(t *testing.T) {
	theta := 7.5
	AC := NewCopula("joe", theta)

	checkTitle("Checking sampling...")
	M := AC.Sample(9000, 3)
	result := AC.Fit(M)

	if math.Abs(AC.theta-theta) > 0.15 {
		t.Errorf("Bad MLE fit, expected theta* = %.3f, got %.3f", theta, AC.theta)
		testERROR()
		fmt.Println(result)
	} else {
		SaveCSV(M, joeSample, ',')
		testOK()
	}
	checkMargins(joeSample)
}

func TestJoeMLE(t *testing.T) {
	M, err := LoadCSV(joeSample, ',', false)
	if err != nil {
		t.Fatal(err)
	}

	checkTitle("Checking MLE fit...")
	AC := NewCopula("joe", -2.0)
	result := AC.Fit(M)
	llFit := result.LogLikelihood

	if math.Abs(AC.theta-7.5) > 0.15 {
		t.Errorf("Bad MLE fit, expected theta* = 7.5, got %f", AC.theta)
		testERROR()
		fmt.Println(result)
	} else {
		testOK()
	}

	checkTitle("Checking max likelihood...")
	ll := AC.LogLikelihood(M)
	if math.Abs(llFit-ll) > 1e-2 {
		t.Errorf("Bad likelihood computation, expected ll = %f, got %f", llFit, -ll)
		testERROR()
	} else {
		testOK()
	}
}

func customRadialPPF(arch *ArchimedeanCopula, p float64, dim int) float64 {
	c := 1.2
	if p > 0. && p < 1. {
		fun := func(z float64, args interface{}) float64 {
			return arch.RadialCdf(math.Pow(z, c), dim) - p
		}

		// fun2 := func(z float64, args interface{}) float64 {
		// 	f := fun(z, args)
		// 	return f * f
		// }

		a := 0.
		b := 1.
		for fun(b, nil) < 0. {
			a = b
			b = 2. * b
		}
		// we know that the cdf is an increasing function
		// so using the bisection algorithm seems enough
		// to find the right quantile
		var tol float64
		switch arch.Family() {
		case "Joe":
			tol = 1e-13
		case "Clayton":
			tol = 1e-6
		default:
			tol = 1e-8
		}
		// root, _, _, err := BFGS(fun2, nil, 0.5*(a+b))
		// root, err := Bisection(fun, nil, a, b, tol)
		// root, err := FalsePosition(fun, nil, a, b, tol)
		root, err := BrentRootFinder(fun, nil, a, b, tol)
		if err != nil {
			fmt.Println(err)
			return -1.
		}
		return math.Pow(root, c)
	}
	return -1.
}

// func TestSolveJoeSampleProblem(t *testing.T) {
// 	size := 50000
// 	dim := 3
// 	theta := 9.
// 	arch := NewCopula("Joe", theta)
// 	M := mat.NewDense(size, dim, nil)

// 	// V := mat.NewDense(size, 2, nil)
// 	// r := make([]float64, 0)

// 	U := uniformSample(size)
// 	for i := 0; i < size; i++ {
// 		Y := standardExpSample(dim)
// 		Sd := scalarDiv(Y, sum(Y))
// 		R := customRadialPPF(arch, U[i], dim)
// 		// R := arch.RadialPpf(U[i], dim)
// 		for R < 0. {
// 			fmt.Println("U = ", U[i])
// 			// R = arch.RadialPpf(rand.Float64(), dim)
// 			R = customRadialPPF(arch, U[i], dim)
// 		}
// 		// V.Set(i, 0, U[i])
// 		// V.Set(i, 1, R)
// 		// r = append(r, R)
// 		for j := 0; j < dim; j++ {
// 			M.Set(i, j, arch.copula.Psi(R*Sd[j], arch.theta))
// 		}
// 	}
// 	path := fmt.Sprintf("resources/joe_%.2f.csv", theta)
// 	SaveCSV(M, path, ',')
// 	checkMargins(path)
// 	// file := fmt.Sprintf("resources/rad_vs_unif_joe_%d_%d.csv", size, dim)
// 	// SaveCSV(V, file, ',')

// }

// func TestSolveJoeSampleOptimizer(t *testing.T) {
// 	dim := 3
// 	arch := NewCopula("Joe", 7.5)

// 	npts := 10000
// 	xmax := 50.
// 	xmin := 1e-15
// 	fun := func(x float64) float64 {
// 		return arch.RadialCdf(x, dim)
// 	}

// 	M := mat.NewDense(npts, 2, nil)
// 	step := (xmax - xmin) / float64(npts)
// 	x := xmin
// 	for i := 0; i < npts; i++ {
// 		M.Set(i, 0, x)
// 		M.Set(i, 1, fun(x))
// 		x += step
// 	}
// 	SaveCSV(M, "resources/joe_radial_cdf.csv", ',')
// }
