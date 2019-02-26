// archimedean.go

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
	// Iterations id the number of iterations of th Brent minimizer
	Iterations int
}

func (fr *FitResult) String() string {
	return fmt.Sprintf("𝜃* %.3f \t[%.3f %.3f]\n\u2113  %.3f",
		fr.Theta,
		fr.LowerBound,
		fr.UpperBound,
		fr.LogLikelihood)
}

// ArchimedeanCopulaFamily is a generic structure defining
// a archimedean copula
type ArchimedeanCopulaFamily struct {
	theta  float64 // the parameter of the generator family
	copula ArchimedeanCopula
}

// ArchimedeanCopula is an interface to implement
// archimedean copula
type ArchimedeanCopula interface {
	ThetaBounds() (float64, float64)
	Psi(t float64, theta float64) float64
	PsiInv(t float64, theta float64) float64
	PsiD(d int, t float64, theta float64) float64
	Cdf(vector []float64, theta float64) float64
	Pdf(vector []float64, theta float64) float64
	LogPdf(vector []float64, theta float64) float64
}

// NewCopula returns a new copula according to the desired family
func NewCopula(family string, theta float64) *ArchimedeanCopulaFamily {
	switch family {
	case "clayton", "Clayton":
		cop := &Clayton{}
		if theta > 0. {
			return &ArchimedeanCopulaFamily{theta: theta, copula: cop}
		}
		return &ArchimedeanCopulaFamily{theta: 1., copula: cop}
	case "joe", "Joe":
		cop := &Joe{}
		if theta >= 1. {
			return &ArchimedeanCopulaFamily{theta: theta, copula: cop}
		}
		return &ArchimedeanCopulaFamily{theta: 1., copula: cop}
	case "frank", "Frank":
		cop := &Frank{}
		if theta > 0. {
			return &ArchimedeanCopulaFamily{theta: theta, copula: cop}
		}
		return &ArchimedeanCopulaFamily{theta: 1., copula: cop}
	case "amh", "AMH":
		cop := &AMH{}
		if theta > 0. && theta < 1. {
			return &ArchimedeanCopulaFamily{theta: theta, copula: cop}
		}
		return &ArchimedeanCopulaFamily{theta: 0.5, copula: cop}
	case "gumbel", "Gumbel":
		cop := &Gumbel{}
		if theta >= 1. {
			return &ArchimedeanCopulaFamily{theta: theta, copula: cop}
		}
		return &ArchimedeanCopulaFamily{theta: 2., copula: cop}
	default:
		return nil
	}
}

// Cdf computes the cumulative distribution function
// of the copula
func (arch *ArchimedeanCopulaFamily) Cdf(vector []float64) float64 {
	return arch.copula.Cdf(vector, arch.theta)
}

// Pdf computes the density of the generated copula
func (arch *ArchimedeanCopulaFamily) Pdf(vector []float64) float64 {
	return arch.copula.Pdf(vector, arch.theta)
}

// LogPdf computes the log density of the generated copula
func (arch *ArchimedeanCopulaFamily) LogPdf(vector []float64) float64 {
	return arch.copula.LogPdf(vector, arch.theta)
}

// LogLikelihood computes the log-likelihood of a batch of
// observations given the underlying archimedean copula
func (arch *ArchimedeanCopulaFamily) LogLikelihood(M *mat.Dense) float64 {
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
func (arch *ArchimedeanCopulaFamily) ConfidenceBounds(M *mat.Dense, level float64) (float64, float64) {
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

func (arch *ArchimedeanCopulaFamily) logLikelihoodToMinimize(theta float64, args interface{}) float64 {
	// arch.theta = theta
	M := args.(*mat.Dense)
	nObs, _ := M.Dims()
	ll := 0.
	for i := 0; i < nObs; i++ {
		// ll += arch.LogPdf(M.RawRowView(i))
		ll += arch.copula.LogPdf(M.RawRowView(i), theta)
	}
	return -ll
}

// Fit estimates the best theta parameter through maximum likelihood
// estimation according to the input observations
func (arch *ArchimedeanCopulaFamily) Fit(M *mat.Dense) *FitResult {
	a, b := arch.copula.ThetaBounds()
	thetaBest, llhood, nit := BrentMinimizer(arch.logLikelihoodToMinimize, M, a, b, 1e-4)
	arch.theta = thetaBest
	down, up := arch.ConfidenceBounds(M, 0.95)
	return &FitResult{
		Theta:         thetaBest,
		LogLikelihood: -llhood,
		UpperBound:    up,
		LowerBound:    down,
		Iterations:    nit}
}

// RadialCdf computes the cdf of the radial part of the ArchimeanCopula
func (arch *ArchimedeanCopulaFamily) RadialCdf(x float64, dim int) float64 {
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
func (arch *ArchimedeanCopulaFamily) RadialPpf(p float64, dim int) float64 {
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
		// fmt.Println(a, b)
		root, err := Bisection(fun, nil, a, b, 5e-7)
		// root, err := BrentRootFinder(fun, nil, a, b, 1e-10)
		if err != nil {
			return -1.
		}
		return root
	}
	return -1.
}

// Sample generates random numbers according to the underlying copula
func (arch *ArchimedeanCopulaFamily) Sample(size int, dim int) *mat.Dense {
	M := mat.NewDense(size, dim, nil)
	rand.Seed(time.Now().UnixNano())
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
