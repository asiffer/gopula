// gumbel_test.go

package gopula

import (
	"fmt"
	"math"
	"testing"
)

var (
	gumbelSample = "resources/gumbel_3.50.csv"
)

func TestInitGumbel(t *testing.T) {
	title("Gumbel")
}

func TestGumbelDistribution(t *testing.T) {
	AC := NewCopula("gumbel", 2.0)

	checkTitle("Checking pdf...")
	pdf := AC.Pdf([]float64{0.5, 0.5})
	if math.Abs(pdf-1.516) > 1e-3 {
		t.Errorf("Bad pdf computation, expected 1.516, got %f", pdf)
		testERROR()
	} else {
		testOK()
	}

	checkTitle("Checking cdf...")
	AC.theta = 2.5
	cdf := AC.Cdf([]float64{0.25, 0.75})
	if math.Abs(cdf-0.2473) > 1e-4 {
		t.Errorf("Bad cdf computation, expected 0.2473, got %f", cdf)
		testERROR()
	} else {
		testOK()
	}

	checkTitle("Checking logpdf...")
	AC.theta = 5.4
	lpdf := AC.LogPdf([]float64{0.35, 0.85})
	if math.Abs(lpdf+6.399161) > 1e-5 {
		t.Errorf("Bad logpdf computation, expected -6.399161, got %f", lpdf)
		testERROR()
	} else {
		testOK()
	}
}

func TestGumbelRadialCdf(t *testing.T) {
	checkTitle("Checking radial cdf...")
	theta := 1.45
	AC := NewCopula("gumbel", theta)
	rcdf := AC.RadialCdf(0.5, 3)
	if math.Abs(rcdf-0.147170) > 1e-5 {
		t.Errorf("Bad radial cdf computation, expected cdf = 0.147170, got %f", rcdf)
		testERROR()
	} else {
		testOK()
	}
}

func TestGumbelRadialPpf(t *testing.T) {
	theta := 2.10
	AC := NewCopula("gumbel", theta)
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

func TestGumbelSampling(t *testing.T) {
	theta := 3.5
	AC := NewCopula("gumbel", theta)

	checkTitle("Checking sampling...")
	M := AC.Sample(9000, 3)
	result := AC.Fit(M)

	if math.Abs(AC.theta-theta) > 0.1 {
		t.Errorf("Bad MLE fit, expected theta* = %.3f, got %.3f", theta, AC.theta)
		testERROR()
		fmt.Println(result)
	} else {
		SaveCSV(M, gumbelSample, ',')
		testOK()
	}
}

func TestGumbelMLE(t *testing.T) {
	M, err := LoadCSV(gumbelSample, ',', false)
	if err != nil {
		t.Fatal(err)
	}

	checkTitle("Checking MLE fit...")
	AC := NewCopula("gumbel", -2.0)
	result := AC.Fit(M)
	llFit := result.LogLikelihood
	if math.Abs(AC.theta-3.5) > 0.15 {
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
