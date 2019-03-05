// gumbel.go

package gopula

import (
	"math"
)

// Gumbel defines the Gumbel copula
type Gumbel struct{}

// Family returns the name of the copula family
func (c *Gumbel) Family() string {
	return "Gumbel"
}

// ThetaBounds returns the range where the copula is well defined
func (c *Gumbel) ThetaBounds() (float64, float64) {
	return 1., Inf
}

// Psi is the generating function of the copula
func (c *Gumbel) Psi(t float64, theta float64) float64 {
	return math.Exp(-math.Pow(t, 1./theta))
}

// PsiInv is the inverse of the generating function of the copula
func (c *Gumbel) PsiInv(t float64, theta float64) float64 {
	return math.Pow(-math.Log(t), theta)
}

func gumbelCoeff(dim int, k int, alpha float64) float64 {
	s := 0.
	a := math.Pow(alpha, float64(k))
	for j := k; j <= dim; j++ {
		s += a * stirlingFirstKind.At(dim, j) * stirlingSecondKind.At(j, k)
		a = a * alpha
	}
	if (dim-k)%2 == 0 {
		return s
	}
	return -s
}

func gumbelPolynom(x float64, dim int, alpha float64) float64 {
	p := 0.
	xk := x
	for k := 1; k <= dim; k++ {
		p += gumbelCoeff(dim, k, alpha) * xk
		xk = xk * x
	}
	return p
}

// PsiD is the d-th derivative of Psi
func (c *Gumbel) PsiD(dim int, t float64, theta float64) float64 {
	coeff := 1.
	alpha := 1. / theta
	if dim%2 == 1 {
		coeff = -1.
	}
	dimF := float64(dim)
	talpha := math.Pow(t, alpha)

	return coeff * c.Psi(t, theta) * gumbelPolynom(talpha, dim, alpha) / math.Pow(t, dimF)
}

// t computes  PsiInv(u_1) + PsiInv(u_2) ... + PsiInv(u_d)
func (c *Gumbel) t(vector []float64, theta float64) float64 {
	sum := 0.
	for _, x := range vector {
		sum += c.PsiInv(x, theta)
	}
	return sum
}

// Cdf computes the cumulative distribution function
// of the copula
func (c *Gumbel) Cdf(vector []float64, theta float64) float64 {
	return c.Psi(c.t(vector, theta), theta)
}

// Pdf computes the density of the generated copula
func (c *Gumbel) Pdf(vector []float64, theta float64) float64 {
	if min(vector) == 0. {
		return 0.
	}
	dim := len(vector)
	dimF := float64(dim)
	alpha := 1. / theta

	tu := c.t(vector, theta)
	tualpha := math.Pow(tu, alpha)

	p1 := math.Pow(theta/tu, dimF) * math.Exp(-tualpha)
	p2 := 1.
	for j := 0; j < dim; j++ {
		p2 = p2 * math.Pow(-math.Log(vector[j]), theta-1.) / vector[j]
	}
	return p1 * p2 * gumbelPolynom(tualpha, dim, alpha)
}

// LogPdf computes the logarithm of the
// density of the copula
func (c *Gumbel) LogPdf(vector []float64, theta float64) float64 {
	if min(vector) == 0. {
		return math.Inf(-1)
	}
	dim := len(vector)
	dimF := float64(dim)
	alpha := 1. / theta

	tu := c.t(vector, theta)
	tualpha := math.Pow(tu, alpha)

	s1 := dimF * math.Log(theta/tu)
	s2 := tualpha

	lvec := log(vector)
	s3 := 0.
	for j := 0; j < dim; j++ {
		s3 += (theta-1.)*math.Log(-lvec[j]) - lvec[j]
	}
	s4 := math.Log(gumbelPolynom(tualpha, dim, alpha))
	return s1 - s2 + s3 + s4
}
