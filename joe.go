// joe.go

package gopula

import (
	"fmt"
	"math"
)

// Joe defines the Joe copula
type Joe struct{}

// Family returns the name of the copula family
func (c *Joe) Family() string {
	return "Joe"
}

// ThetaBounds returns the range where the copula is well defined
func (c *Joe) ThetaBounds() (float64, float64) {
	return 1., Inf
}

// Psi is the generating function of the copula
func (c *Joe) Psi(t float64, theta float64) float64 {
	return 1. - math.Pow(1.-math.Exp(-t), 1./theta)
}

// PsiInv is the inverse of the generating function of the copula
func (c *Joe) PsiInv(t float64, theta float64) float64 {
	return -math.Log(1 - math.Pow(1.-t, theta))
}

func joeCoeff(dim int, k int, alpha float64) float64 {
	return stirlingSecondKind.At(dim, k+1) * math.Gamma(float64(k+1)-alpha) / math.Gamma(1.-alpha)
}

func joePolynom(x float64, dim int, alpha float64) float64 {
	p := 0.
	xk := 1.
	for k := 0; k < dim; k++ {
		p += joeCoeff(dim, k, alpha) * xk
		xk = xk * x
	}
	return p
}

func joeH(vector []float64, dim int, theta float64) float64 {
	h := 1.
	for j := 0; j < dim; j++ {
		h = h * (1 - math.Pow(1-vector[j], theta))
	}
	return h
}

// PsiD is the d-th derivative of Psi
func (c *Joe) PsiD(d int, t float64, theta float64) float64 {
	coeff := 1.
	alpha := 1. / theta
	if d%2 == 1 {
		coeff = -1.
	}
	e := math.Exp(-t)
	return coeff * alpha * e * joePolynom(e/(1.-e), d, alpha) / math.Pow(1.-e, 1.-alpha)
}

// t computes  PsiInv(u_1) + PsiInv(u_2) ... + PsiInv(u_d)
func (c *Joe) t(vector []float64, theta float64) float64 {
	sum := 0.
	for _, x := range vector {
		sum += c.PsiInv(x, theta)
	}
	return sum
}

// Cdf computes the cumulative distribution function
// of the copula
func (c *Joe) Cdf(vector []float64, theta float64) float64 {
	return c.Psi(c.t(vector, theta), theta)
}

// Pdf computes the density of the generated copula
func (c *Joe) Pdf(vector []float64, theta float64) float64 {
	if min(vector) == 0. {
		return 0.
	}

	dim := len(vector)
	dimF := float64(dim)
	alpha := 1. / theta

	if min(vector) >= 0.96 {
		epsilon := 1 - mean(vector)
		return math.Pow(theta, dimF-1.) *
			math.Pow(dimF, alpha-dimF) *
			math.Pow(epsilon, 1-dimF)
	}

	h := 1.
	num := 1.
	for j := 0; j < dim; j++ {
		h = h * (1 - math.Pow(1-vector[j], theta))
		num = num * math.Pow(1-vector[j], theta-1)
	}
	return math.Pow(theta, dimF-1.) * num * joePolynom(h/(1.-h), dim, alpha) / math.Pow(1.-h, 1.-alpha)
}

// LogPdf computes the logarithm of the
// density of the copula
func (c *Joe) LogPdf(vector []float64, theta float64) float64 {
	if min(vector) == 0. {
		return math.Inf(-1)
	}
	dim := len(vector)
	dimF := float64(dim)
	alpha := 1. / theta

	if min(vector) >= 0.96 {
		epsilon := 1 - mean(vector)
		return (dimF-1)*math.Log(theta) -
			(dimF-alpha)*math.Log(dimF) -
			(dimF-1)*math.Log(epsilon)
	}

	s1 := (dimF - 1) * math.Log(theta)
	s2 := 0.

	h := 1.
	for j := 0; j < dim; j++ {
		s2 += math.Log(1 - vector[j])
		h = h * (1 - math.Pow(1-vector[j], theta))
	}
	s2 = (theta - 1) * s2

	s3 := (1 - alpha) * math.Log(1-h)
	s4 := math.Log(joePolynom(h/(1-h), dim, alpha))

	if math.IsInf(s3, 1) || math.IsInf(s3, -1) {
		fmt.Println(s4, h, vector)
	}

	return s1 + s2 - s3 + s4
}
