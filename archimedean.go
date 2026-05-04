// archimedean.go

// Package gopula implements common Archimedean Copulas. It aims
// both to infer a copula from observations and to sample data from
// a given model.
package gopula

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

var (
	stirlingFirstKind = []float64{
		1,    // s(0,0)
		0, 1, // s(1,0), s(1,1)
		0, -1, 1,
		0, 2, -3, 1,
		0, -6, 11, -6, 1,
		0, 24, -50, 35, -10, 1,
		0, -120, 274, -225, 85, -15, 1,
		0, 720, -1764, 1624, -735, 175, -21, 1,
		0, -5040, 13068, -13132, 6769, -1960, 322, -28, 1,
		0, 40320, -109584, 118124, -67284, 22449, -4536, 546, -36, 1,
		0, -362880, 1026576, -1172700, 723680, -269325, 63273, -9450, 870, -45, 1,
		0, 3628800, -10628640, 12753576, -8409500, 3416930, -902055, 157773, -18150, 1320, -55, 1,
		0, -39916800, 120543840, -150917976, 105258076, -45995730, 13339535, -2637558, 357423, -32670, 1925, -66, 1,
		0, 479001600, -1486442880, 1931559552, -1414014888, 657206836, -206070150, 44990231, -6926634, 749463, -55770, 2717, -78, 1,
		0, -6227020800, 19802759040, -26596717056, 20313753096, -9957703756, 3336118786, -790943153, 135036473, -16669653, 1474473, -91091, 3731, -91, 1,
		0, 87178291200, -283465647360, 392156797824, -310989260400, 159721605680, -56663366760, 14409322928, -2681453775, 368411615, -37312275, 2749747, -143325, 5005, -105, 1,
		0, -1307674368000, 4339163001600, -6165817614720, 5056995703824, -2706813345600, 1009672107080, -272803210680, 54631129553, -8207628000, 928095740, -78558480, 4899622, -218400, 6580, -120, 1,
		0, 20922789888000, -70734282393600, 102992244837120, -87077748875904, 48366009233424, -18861567058880, 5374523477960, -1146901283528, 185953177553, -23057159840, 2185031420, -156952432, 8394022, -323680, 8500, -136, 1,
		0, -355687428096000, 1223405590579200, -1821602444624640, 1583313975727488, -909299905844112, 369012649234384, -110228466184200, 24871845297936, -4308105301929, 577924894833, -60202693980, 4853222764, -299650806, 13896582, -468180, 10812, -153, 1,
		0, 6402373705728000, -22376988058521600, 34012249593822720, -30321254007719424, 17950712280921504, -7551527592063024, 2353125040549984, -557921681547048, 102417740732658, -14710753408923, 1661573386473, -147560703732, 10246937272, -549789282, 22323822, -662796, 13566, -171, 1,
		0, -121645100408832000, 431565146817638400, -668609730341153280, 610116075740491776, -371384787345228000, 161429736530118960, -52260903362512720, 12953636989943896, -2503858755467550, 381922055502195, -46280647751910, 4465226757381, -342252511900, 20692933630, -973941900, 34916946, -920550, 16815, -190, 1,
	}
	maxDim1            = 20
	stirlingSecondKind = []float64{
		1,    // S(0,0)
		0, 1, // S(1,0), S(1,1)
		0, 1, 1,
		0, 1, 3, 1,
		0, 1, 7, 6, 1,
		0, 1, 15, 25, 10, 1,
		0, 1, 31, 90, 65, 15, 1,
		0, 1, 63, 301, 350, 140, 21, 1,
		0, 1, 127, 966, 1701, 1050, 266, 28, 1,
		0, 1, 255, 3025, 7770, 6951, 2646, 462, 36, 1,
		0, 1, 511, 9330, 34105, 42525, 22827, 5880, 750, 45, 1,
		0, 1, 1023, 28501, 145750, 246730, 179487, 63987, 11880, 1155, 55, 1,
		0, 1, 2047, 86526, 611501, 1379400, 1323652, 627396, 159027, 22275, 1705, 66, 1,
		0, 1, 4095, 261625, 2532530, 7508501, 9321312, 5715424, 1899612, 359502, 39325, 2431, 78, 1,
		0, 1, 8191, 788970, 10391745, 40075035, 63436373, 49329280, 20912320, 5135130, 752752, 66066, 3367, 91, 1,
		0, 1, 16383, 2375101, 42355950, 210766920, 420693273, 408741333, 216627840, 67128490, 12662650, 1479478, 106470, 4550, 105, 1,
		0, 1, 32767, 7141686, 171798901, 1096190550, 2734926558, 3281882604, 2141764053, 820784250, 193754990, 28936908, 2757118, 165620, 6020, 120, 1,
		0, 1, 65535, 21457825, 694337290, 5652751651, 17505749898, 25708104786, 20415995028, 9528822303, 2758334150, 512060978, 62022324, 4910178, 249900, 7820, 136, 1,
		0, 1, 131071, 64439010, 2798806985, 28958095545, 110687251039, 197462483400, 189036065010, 106175395755, 37112163803, 8391004908, 1256328866, 125854638, 8408778, 367200, 9996, 153, 1,
		0, 1, 262143, 193448101, 11259666950, 147589284710, 693081601779, 1492924634839, 1709751003480, 1144614626805, 477297033785, 129413217791, 23466951300, 2892439160, 243577530, 13916778, 527136, 12597, 171, 1,
		0, 1, 524287, 580606446, 45232115901, 749206090500, 4306078895384, 11143554045652, 15170932662679, 12011282644725, 5917584964655, 1900842429486, 411016633391, 61068660380, 6302524580, 452329200, 22350954, 741285, 15675, 190, 1,
	}
	maxDim2 = 20
)

var (
	inf = 25.0 // big values to approxiate Infinity
)

// SetInf sets the value of 'infinity' used in the package.
// This is used to approximate mathematical infinity in
// functions like ThetaBounds.
// The default value is 25.0, which should be sufficiently
// large for most practical purposes, but can be adjusted if needed.
func SetInf(value float64) {
	inf = value
}

func ComputeStirling1Until(n int) {
	for i := maxDim1; i <= n; i++ {
		for j := 1; j <= i; j++ {
			sfk := stirling1(i-1, j-1) - float64(i-1)*stirling1(i-1, j)
			stirlingFirstKind = append(stirlingFirstKind, sfk)
		}
	}
	maxDim1 = n
}

func ComputeStirling2Until(n int) {
	for i := maxDim2; i <= n; i++ {
		for j := 1; j <= i; j++ {
			ssk := stirling2(i-1, j-1) + float64(j)*stirling2(i-1, j)
			stirlingSecondKind = append(stirlingSecondKind, ssk)
		}
	}
	maxDim2 = n
}

func stirling2(n int, k int) float64 {
	if n < 0 || k < 0 || k > n {
		return 0.
	}
	if n > maxDim2 {
		ComputeStirling2Until(n)
		return stirling2(n, k)
	}
	return stirlingSecondKind[k+n*(n+1)/2]
}

func stirling1(n int, k int) float64 {
	if n < 0 || k < 0 || k > n {
		return 0.
	}
	if n > maxDim1 {
		ComputeStirling1Until(n)
		return stirling1(n, k)
	}
	return stirlingFirstKind[k+n*(n+1)/2]
}

// var (
// 	// MaxDim is the maximum dimension for which
// 	// stirling number are pre-computed
// 	MaxDim             = 12
// 	stirlingFirstKind  = mat.NewDense(MaxDim, MaxDim, nil)
// 	stirlingSecondKind = mat.NewDense(MaxDim, MaxDim, nil)
// 	// Inf is a 'big' value (for optimizing bound purpose)
// 	Inf = 12.0
// )

// func init() {
// 	PrecomputeStirlingNumbers()
// }

// // PrecomputeStirlingNumbers computes first and second kind stirling numbers
// // until MaxDim
// func PrecomputeStirlingNumbers() {
// 	stirlingFirstKind.Set(0, 0, 1.)
// 	stirlingSecondKind.Set(0, 0, 1.)
// 	for i := 1; i < MaxDim; i++ {
// 		for j := 1; j < MaxDim; j++ {

// 			sfk := stirlingFirstKind.At(i-1, j-1) - float64(i-1)*stirlingFirstKind.At(i-1, j)
// 			stirlingFirstKind.Set(i, j, sfk)
// 			ssk := stirlingSecondKind.At(i-1, j-1) + float64(j)*stirlingSecondKind.At(i-1, j)
// 			stirlingSecondKind.Set(i, j, ssk)

// 		}
// 	}
// }

// FitResult is a basic structure detailing the output of the fit
type FitResult struct {
	// Theta is the estimated parameter
	Theta float64
	// LogLikelihood is the corresponding log-likelihood (the maximum)
	LogLikelihood float64
	// UpperBound is the 95% upper confidence bound
	UpperBound float64
	// LowerBound is the 95% lower confidence bound
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
	Family() string
	ThetaBounds() (float64, float64)
	Psi(t float64, theta float64) float64
	PsiInv(t float64, theta float64) float64
	PsiD(d int, t float64, theta float64) float64
	Cdf(vector []float64, theta float64) float64
	Pdf(vector []float64, theta float64) float64
	LogPdf(vector []float64, theta float64) float64
}

// NewCopula returns a new copula for the given family and theta parameter.
// Returns ErrUnknownFamily for unrecognised family names and
// ErrThetaOutOfBounds when theta violates the family's mathematical constraints.
func NewCopula(family string, theta float64) (*ArchimedeanCopula, error) {
	switch family {
	case "clayton", "Clayton":
		if theta <= 0 {
			return nil, fmt.Errorf("%w: Clayton requires theta > 0, got %g", ErrThetaOutOfBounds, theta)
		}
		return &ArchimedeanCopula{theta: theta, copula: &Clayton{}}, nil
	case "joe", "Joe":
		if theta < 1 {
			return nil, fmt.Errorf("%w: Joe requires theta >= 1, got %g", ErrThetaOutOfBounds, theta)
		}
		return &ArchimedeanCopula{theta: theta, copula: &Joe{}}, nil
	case "frank", "Frank":
		if theta <= 0 {
			return nil, fmt.Errorf("%w: Frank requires theta > 0, got %g", ErrThetaOutOfBounds, theta)
		}
		return &ArchimedeanCopula{theta: theta, copula: &Frank{}}, nil
	case "amh", "AMH":
		if theta < 0 || theta >= 1 {
			return nil, fmt.Errorf("%w: AMH requires theta in (0, 1), got %g", ErrThetaOutOfBounds, theta)
		}
		return &ArchimedeanCopula{theta: theta, copula: &AMH{}}, nil
	case "gumbel", "Gumbel":
		if theta < 1 {
			return nil, fmt.Errorf("%w: Gumbel requires theta >= 1, got %g", ErrThetaOutOfBounds, theta)
		}
		return &ArchimedeanCopula{theta: theta, copula: &Gumbel{}}, nil
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnknownFamily, family)
	}
}

// Family returns the name of the copula family
func (arch *ArchimedeanCopula) Family() string {
	return arch.copula.Family()
}

// Theta returns the current value of 𝜃
func (arch *ArchimedeanCopula) Theta() float64 {
	return arch.theta
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

// Validate checks that all elements of vector are strictly in (0, 1),
// as required by the copula density and distribution functions.
func (arch *ArchimedeanCopula) Validate(vector []float64) error {
	for i, x := range vector {
		if x <= 0 || x >= 1 {
			return fmt.Errorf("%w: element %d is %g", ErrInvalidInput, i, x)
		}
	}
	return nil
}

// LogLikelihood computes the log-likelihood of a batch of
// observations given the underlying archimedean copula
func (arch *ArchimedeanCopula) LogLikelihood(M *mat.Dense) float64 {
	nObs, _ := M.Dims()
	ll := 0.
	for i := range nObs {
		lpdf := arch.LogPdf(M.RawRowView(i))
		if !math.IsNaN(lpdf) {
			ll += lpdf
		}
	}
	return ll
}

// ConfidenceBounds computes the upper and lower confidence bounds
// at given level (level = 1-alpha = 0.95 in practice). The parameter
// theta must be the fitted value.
func (arch *ArchimedeanCopula) ConfidenceBounds(M *mat.Dense, level float64) (float64, float64, error) {
	ll := arch.LogLikelihood(M)
	cs := distuv.ChiSquared{K: 1}
	q := cs.Quantile(level)
	fun := func(x float64, args *mat.Dense) float64 {
		return arch.logLikelihoodToMinimize(x, args) + (ll - q/2)
	}
	maxDown, maxUp := arch.copula.ThetaBounds()
	if math.IsInf(maxUp, 1) {
		maxUp = math.Max(2*arch.theta, inf)
	}
	thetaUp, errUp := Bisection(fun, M, arch.theta, maxUp, 1e-6)
	thetaDown, errDown := Bisection(fun, M, maxDown, arch.theta, 1e-6)
	if errUp != nil {
		return thetaDown, thetaUp, fmt.Errorf("upper bound: %w", errUp)
	}
	if errDown != nil {
		return thetaDown, thetaUp, fmt.Errorf("lower bound: %w", errDown)
	}
	return thetaDown, thetaUp, nil
}

func (arch *ArchimedeanCopula) logLikelihoodToMinimize(theta float64, M *mat.Dense) float64 {
	// the argument is casted to a matrix
	// M := args.(*mat.Dense)
	nObs, _ := M.Dims()
	ll := 0.
	for i := range nObs {
		lpdf := arch.copula.LogPdf(M.RawRowView(i), theta)
		if !math.IsNaN(lpdf) {
			ll += lpdf
		}
	}
	return -ll
}

// Fit estimates the best theta parameter through maximum likelihood
// estimation according to the input observations.
func (arch *ArchimedeanCopula) Fit(M *mat.Dense) (*FitResult, error) {
	msg := ""
	a, b := arch.copula.ThetaBounds()
	if math.IsInf(b, 1) {
		b = inf
	}
	thetaBest, llhood, feval, optimErr := BrentMinimizer(arch.logLikelihoodToMinimize, M, a+1e-8, b, 1e-8)
	if math.Min(math.Abs(thetaBest-a), math.Abs(thetaBest-b)) < 1e-2 || optimErr != nil {
		if optimErr != nil {
			msg += "Brent minimizer failed: " + optimErr.Error() + ". "
		} else {
			msg += "Brent minimizer hit the bounds. "
		}
		msg += "Falling back to BFGS. "

		x0 := 0.5 * (a + b)
		thetaBest, llhood, feval, optimErr = BFGS(arch.logLikelihoodToMinimize, M, x0)
	}
	// thetaBest, llhood, feval, optimErr := BFGS(arch.logLikelihoodToMinimize, M, x0)
	if optimErr != nil {
		msg += "Error: " + optimErr.Error()
	} else {
		msg += "Success"
	}
	arch.theta = thetaBest
	down, up, boundsErr := arch.ConfidenceBounds(M, 0.95)
	if boundsErr != nil {
		msg += "; confidence bounds: " + boundsErr.Error()
	}
	return &FitResult{
		Theta:         thetaBest,
		LogLikelihood: -llhood,
		UpperBound:    up,
		LowerBound:    down,
		Evals:         feval,
		Message:       msg,
	}, optimErr
}

// RadialCdf computes the cdf of the radial part of the ArchimeanCopula
func (arch *ArchimedeanCopula) RadialCdf(x float64, dim int) float64 {
	if x <= 0. {
		return 0.
	}

	cdfx := 1. - arch.copula.Psi(x, arch.theta)
	f := 1.0
	for k := 1; k <= dim-1; k++ {
		f = f * (-x) / float64(k)
		cdfx = cdfx - f*arch.copula.PsiD(k, x, arch.theta)
	}
	return cdfx
}

// RadialPpf computes the quantile zp verifying P(X<zp) = p
func (arch *ArchimedeanCopula) RadialPpf(p float64, dim int) float64 {
	c := 0.95
	if p > 0. && p < 1. {
		var tol float64
		switch arch.Family() {
		case "Joe":
			tol = 1e-10
			c = 1.25
		case "Clayton":
			tol = 1e-6
		default:
			tol = 1e-8
		}

		fun := func(z float64, M *mat.Dense) float64 {
			return arch.RadialCdf(math.Pow(z, c), dim) - p
		}

		a := 0.
		b := 0.2
		for fun(b, nil) < 0. {
			a = b
			b = 2. * b
		}

		root, err := Bisection(fun, nil, a, b, tol)
		if err != nil {
			fmt.Println(err)
			return -1.
		}
		return math.Pow(root, c)
	}
	return -1.
}

// Sample generates random numbers according to the underlying copula.
// Returns ErrSamplingFailed if the radial PPF cannot converge for any point.
func (arch *ArchimedeanCopula) Sample(size int, dim int) (*mat.Dense, error) {
	M := mat.NewDense(size, dim, nil)
	U := uniformSample(size)
	for i := range size {
		Y := standardExpSample(dim)
		Sd := scalarDiv(Y, sum(Y))
		R := arch.RadialPpf(U[i], dim)
		for retries := 0; R < 0. && retries < 100; retries++ {
			R = arch.RadialPpf(rand.Float64(), dim)
		}
		if R < 0. {
			return nil, fmt.Errorf("%w at sample index %d", ErrSamplingFailed, i)
		}
		for j := range dim {
			M.Set(i, j, arch.copula.Psi(R*Sd[j], arch.theta))
		}
	}
	return M, nil
}
