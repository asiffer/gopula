// clayton.go

package gopula

import (
	"math"
)

// Clayton defines the clayton copula
type Clayton struct{}

// ThetaBounds returns the range where the copula is well defined
func (c *Clayton) ThetaBounds() (float64, float64) {
	return 0., Inf
}

// Psi is the generating function of the copula
func (c *Clayton) Psi(t float64, theta float64) float64 {
	return math.Pow(1.+t, -1./theta)
}

// PsiInv is the inverse of the generating function of the copula
func (c *Clayton) PsiInv(t float64, theta float64) float64 {
	return math.Pow(t, -theta) - 1.
}

// PsiD is the d-th derivative of Psi
func (c *Clayton) PsiD(d int, t float64, theta float64) float64 {
	coeff := 1.
	alpha := 1. / theta
	if d%2 == 1 {
		coeff = -1.
	}
	df := float64(d)
	return coeff * math.Pow(1.+t, -alpha-df) * math.Gamma(df+alpha) / math.Gamma(alpha)
}

// t computes  PsiInv(u_1) + PsiInv(u_2) ... + PsiInv(u_d)
func (c *Clayton) t(vector []float64, theta float64) float64 {
	sum := 0.
	for _, x := range vector {
		sum += c.PsiInv(x, theta)
	}
	return sum
}

// Cdf computes the cumulative distribution function
// of the copula
func (c *Clayton) Cdf(vector []float64, theta float64) float64 {
	return c.Psi(c.t(vector, theta), theta)
}

// Pdf computes the density of the generated copula
func (c *Clayton) Pdf(vector []float64, theta float64) float64 {
	if min(vector) == 0. {
		return 0.
	}
	dim := len(vector)
	dimF := float64(dim)
	alpha := 1. / theta
	p1 := 1.
	for i := 0.; i < dimF; i += 1.0 {
		p1 = p1 * (1. + theta*i)
	}
	p2 := math.Pow(prod(vector), -1.-theta)
	p3 := math.Pow(1.+c.t(vector, theta), -dimF-alpha)
	return p1 * p2 * p3
}

// LogPdf computes the logarithm of the
// density of the copula
func (c *Clayton) LogPdf(vector []float64, theta float64) float64 {
	if min(vector) == 0. {
		return math.Inf(-1)
	}
	dim := len(vector)
	dimF := float64(dim)
	alpha := 1. / theta
	s1 := 0.
	for i := 1.0; i < dimF; i += 1.0 {
		s1 += math.Log(1. + theta*i)
	}
	s2 := (1. + theta) * sum(log(vector))
	s3 := (dimF + alpha) * math.Log(1.+c.t(vector, theta))
	return s1 - s2 - s3
}
