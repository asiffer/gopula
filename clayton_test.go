// clayton_test.go

package gopula

import (
	"fmt"
	"math"
	"testing"
)

var (
	claytonSample = "resources/clayton_2.10.csv"
)

func TestInitClayton(t *testing.T) {
	title("Clayton")
}

func TestClaytonDistribution(t *testing.T) {
	C := NewCopula("clayton", 2.0)
	pdf := C.Pdf([]float64{0.5, 0.5})

	checkTitle("Checking pdf...")
	if math.Abs(pdf-1.481) > 1e-3 {
		t.Errorf("Bad pdf computation, expected 1.481, got %f", pdf)
		testERROR()
	} else {
		testOK()
	}

	checkTitle("Checking cdf...")
	C.theta = 0.5
	cdf := C.Cdf([]float64{0.25, 0.75})
	if math.Abs(cdf-0.21539) > 1e-5 {
		t.Errorf("Bad cdf computation, expected 0.21539, got %f", cdf)
		testERROR()
	} else {
		testOK()
	}

	C.theta = 5.4
	checkTitle("Checking logpdf...")
	lpdf := C.LogPdf([]float64{0.35, 0.85})
	if math.Abs(lpdf+2.78319) > 1e-5 {
		t.Errorf("Bad logpdf computation, expected -2.78319, got %f", lpdf)
		testERROR()
	} else {
		testOK()
	}
}

func TestClaytonRadialCdf(t *testing.T) {
	checkTitle("Checking radial cdf...")
	theta := 1.45
	AC := NewCopula("clayton", theta)
	rcdf := AC.RadialCdf(0.5, 3)
	if math.Abs(rcdf-0.02118) > 1e-5 {
		t.Errorf("Bad radial cdf computation, expected cdf = 0.02118, got %f", rcdf)
		testERROR()
	} else {
		testOK()
	}
}

func TestClaytonRadialPpf(t *testing.T) {
	theta := 2.10
	AC := NewCopula("clayton", theta)
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

func TestClaytonSampling(t *testing.T) {
	theta := 2.10
	AC := NewCopula("clayton", theta)

	checkTitle("Checking sampling...")
	M := AC.Sample(9000, 3)
	result := AC.Fit(M)
	if math.Abs(AC.theta-theta) > 0.15 {
		t.Errorf("Bad MLE fit, expected theta* = %.3f, got %.3f", theta, AC.theta)
		testERROR()
		fmt.Println(result)
	} else {
		SaveCSV(M, claytonSample, ',')
		testOK()
	}
}

func TestClaytonMLE(t *testing.T) {
	M, err := LoadCSV(claytonSample, ',', false)
	if err != nil {
		t.Fatal(err)
	}

	checkTitle("Checking MLE fit...")
	AC := NewCopula("clayton", -3.)
	result := AC.Fit(M)
	llFit := result.LogLikelihood
	if math.Abs(AC.theta-2.10) > 0.15 {
		t.Errorf("Bad MLE fit, expected theta* = 5.75, got %f", AC.theta)
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
