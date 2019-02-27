// archimedean.go

// Package gopula implements common Archimedean Copulas. It aims
// both to infer a copula from observations and to sample data from
// a given model.
package gopula

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	// MaxDim is the maximum dimension for which
	// stirling number are pre-computed
	MaxDim             = 12
	stirlingFirstKind  = mat.NewDense(MaxDim, MaxDim, nil)
	stirlingSecondKind = mat.NewDense(MaxDim, MaxDim, nil)
	// Inf is a 'big' value (for optimizing bound purpose)
	Inf = 15.
)

func init() {
	rand.Seed(time.Now().UnixNano())
	PrecomputeStirlingNumbers()
}

// PrecomputeStirlingNumbers computes first and second kind stirling numbers
// until MaxDim
func PrecomputeStirlingNumbers() {
	stirlingFirstKind.Set(0, 0, 1.)
	stirlingSecondKind.Set(0, 0, 1.)
	for i := 1; i < MaxDim; i++ {
		for j := 1; j < MaxDim; j++ {
			// if i == 0 && j == 0 {
			// 	stirlingFirstKind.Set(i, j, 1.)
			// 	stirlingSecondKind.Set(i, j, 1.)
			// } else if i == 0 || j == 0 {
			// 	stirlingFirstKind.Set(i, j, 0.)
			// 	stirlingSecondKind.Set(i, j, 0.)
			// } else {
			sfk := stirlingFirstKind.At(i-1, j-1) - float64(i-1)*stirlingFirstKind.At(i-1, j)
			stirlingFirstKind.Set(i, j, sfk)
			ssk := stirlingSecondKind.At(i-1, j-1) + float64(j)*stirlingSecondKind.At(i-1, j)
			stirlingSecondKind.Set(i, j, ssk)
			// }

		}
	}
}

// FitResult is a basic structure detailing the output of the fit
type FitResult struct {
	// Theta is the estimated parameter
	Theta float64
	// LogLikelihhod is the correspond log-likelihood (the maximum)
	LogLikelihood float64
	// UpperBound is the 95% upper confidence bound
	UpperBound float64
	// LowerBound is the 95% upper confidence bound
	LowerBound float64
	// Evals is the number of function evaluations
	Evals int
	// Message describes whether the fit has suceeded
	Message string
}

func (fr *FitResult) String() string {
	format := "%8s %.6f\n%8s %.6f\n%8s [%.3f, %.3f]\n%8s %d\n%8s %s"
	return fmt.Sprintf(format,
		"ℓ", fr.LogLikelihood,
		"𝜃", fr.Theta,
		"95%", fr.LowerBound, fr.UpperBound,
		"Evals", fr.Evals,
		"Message", fr.Message)
}

// ArchimedeanCopula is a generic structure defining
// an archimedean copula
type ArchimedeanCopula struct {
	theta  float64 // the parameter of the generator family
	copula ArchimedeanCopuler
}

// ArchimedeanCopuler is an interface to implement
// an archimedean copula
type ArchimedeanCopuler interface {
	ThetaBounds() (float64, float64)
	Psi(t float64, theta float64) float64
	PsiInv(t float64, theta float64) float64
	PsiD(d int, t float64, theta float64) float64
	Cdf(vector []float64, theta float64) float64
	Pdf(vector []float64, theta float64) float64
	LogPdf(vector []float64, theta float64) float64
}

// NewCopula returns a new copula according to the desired family
func NewCopula(family string, theta float64) *ArchimedeanCopula {
	switch family {
	case "clayton", "Clayton":
		cop := &Clayton{}
		if theta > 0. {
			return &ArchimedeanCopula{theta: theta, copula: cop}
		}
		return &ArchimedeanCopula{theta: 1., copula: cop}
	case "joe", "Joe":
		cop := &Joe{}
		if theta >= 1. {
			return &ArchimedeanCopula{theta: theta, copula: cop}
		}
		return &ArchimedeanCopula{theta: 1., copula: cop}
	case "frank", "Frank":
		cop := &Frank{}
		if theta > 0. {
			return &ArchimedeanCopula{theta: theta, copula: cop}
		}
		return &ArchimedeanCopula{theta: 1., copula: cop}
	case "amh", "AMH":
		cop := &AMH{}
		if theta > 0. && theta < 1. {
			return &ArchimedeanCopula{theta: theta, copula: cop}
		}
		return &ArchimedeanCopula{theta: 0.5, copula: cop}
	case "gumbel", "Gumbel":
		cop := &Gumbel{}
		if theta >= 1. {
			return &ArchimedeanCopula{theta: theta, copula: cop}
		}
		return &ArchimedeanCopula{theta: 2., copula: cop}
	default:
		return nil
	}
}

// Cdf computes the cumulative distribution function
// of the copula
func (arch *ArchimedeanCopula) Cdf(vector []float64) float64 {
	return arch.copula.Cdf(vector, arch.theta)
}

// Pdf computes the density of the generated copula
func (arch *ArchimedeanCopula) Pdf(vector []float64) float64 {
	return arch.copula.Pdf(vector, arch.theta)
}

// LogPdf computes the log density of the generated copula
func (arch *ArchimedeanCopula) LogPdf(vector []float64) float64 {
	return arch.copula.LogPdf(vector, arch.theta)
}

// LogLikelihood computes the log-likelihood of a batch of
// observations given the underlying archimedean copula
func (arch *ArchimedeanCopula) LogLikelihood(M *mat.Dense) float64 {
	nObs, _ := M.Dims()
	ll := 0.
	for i := 0; i < nObs; i++ {
		ll += arch.LogPdf(M.RawRowView(i))
	}
	return ll
}

// ConfidenceBounds compute the upper and lower confidence bounds
// at given level (level = 1-alpha = 0.95 in practice). The parameter
// theta must be the fitted value.
func (arch *ArchimedeanCopula) ConfidenceBounds(M *mat.Dense, level float64) (float64, float64) {
	ll := arch.LogLikelihood(M)
	cs := distuv.ChiSquared{K: 1, Src: nil}
	q := cs.Quantile(level)
	fun := func(x float64, args interface{}) float64 {
		return arch.logLikelihoodToMinimize(x, M) + (ll - q/2)
	}
	maxDown, maxUp := arch.copula.ThetaBounds()
	thetaUp, _ := Bisection(fun, nil, arch.theta, maxUp, 1e-8)
	thetaDown, _ := Bisection(fun, nil, maxDown, arch.theta, 1e-8)
	return thetaDown, thetaUp
}

func (arch *ArchimedeanCopula) logLikelihoodToMinimize(theta float64, args interface{}) float64 {
	// the argument is casted to a matrix
	M := args.(*mat.Dense)
	nObs, _ := M.Dims()
	ll := 0.
	for i := 0; i < nObs; i++ {
		ll += arch.copula.LogPdf(M.RawRowView(i), theta)
	}
	// we return the opposite of the loglikelihood (for minimization)
	return -ll
}

// Fit estimates the best theta parameter through maximum likelihood
// estimation according to the input observations
func (arch *ArchimedeanCopula) Fit(M *mat.Dense) *FitResult {
	var msg string
	a, b := arch.copula.ThetaBounds()
	thetaBest, llhood, feval, err := BrentMinimizer(arch.logLikelihoodToMinimize, M, a, b, 1e-5)
	if math.Min(math.Abs(thetaBest-a), math.Abs(thetaBest-b)) < 1e-2 {
		msg = "Falling back to BFGS."
		thetaBest, llhood, feval, err = BFGS(arch.logLikelihoodToMinimize, M, 0.5*(a+b))
	}
	if err != nil {
		msg += " Error: " + err.Error()
	} else {
		msg += "Success"
	}
	// thetaBest, llhood, nit := BFGS(arch.logLikelihoodToMinimize, M, 0.5*(a+b))
	arch.theta = thetaBest
	down, up := arch.ConfidenceBounds(M, 0.95)
	return &FitResult{
		Theta:         thetaBest,
		LogLikelihood: -llhood,
		UpperBound:    up,
		LowerBound:    down,
		Evals:         feval,
		Message:       msg}
}

// RadialCdf computes the cdf of the radial part of the ArchimeanCopula
func (arch *ArchimedeanCopula) RadialCdf(x float64, dim int) float64 {
	if x <= 0. {
		return 0.
	}
	cdfx := 1. - arch.copula.Psi(x, arch.theta)
	psid := arch.copula.PsiD(dim-1, x, arch.theta)
	dimF := float64(dim)
	if psid > 0 {
		if dim%2 == 0 {
			cdfx = cdfx + math.Pow(x, dimF-1.)*psid/float64(factorial(dim-1))
		} else {
			cdfx = cdfx - math.Pow(x, dimF-1.)*psid/float64(factorial(dim-1))
		}
	}
	f := 1.0
	for k := 1; k < dim-1; k++ {
		f = f * (-1.0 / float64(k))
		cdfx = cdfx - f*math.Pow(x, float64(k))*arch.copula.PsiD(k, x, arch.theta)
	}
	return cdfx
}

// RadialPpf computes the quantile zp verifying P(X<zp) = p
func (arch *ArchimedeanCopula) RadialPpf(p float64, dim int) float64 {
	if p > 0. && p < 1. {
		fun := func(z float64, args interface{}) float64 {
			return arch.RadialCdf(z, dim) - p
		}
		a := 0.
		b := 0.5
		for fun(b, nil) < 0 {
			a = b
			b = 2 * b
		}
		// we know that the cdf is an increasing function
		// so using the bisection algorithm seems enough
		// to find the right quantile
		root, err := Bisection(fun, nil, a, b, 1e-6)
		if err != nil {
			fmt.Println(err)
			return -1.
		}
		return root
	}
	return -1.
}

// Sample generates random numbers according to the underlying copula
func (arch *ArchimedeanCopula) Sample(size int, dim int) *mat.Dense {
	M := mat.NewDense(size, dim, nil)

	U := uniformSample(size)
	for i := 0; i < size; i++ {
		Y := standardExpSample(dim)
		Sd := scalarDiv(Y, sum(Y))
		R := arch.RadialPpf(U[i], dim)
		if R == 0. {
			fmt.Println(R, Sd, U[i], arch.RadialPpf(U[i], 3))
		}
		for j := 0; j < dim; j++ {
			M.Set(i, j, arch.copula.Psi(R*Sd[j], arch.theta))
		}
	}
	return M
}
