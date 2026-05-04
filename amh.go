// amh.go

package gopula

import (
	"math"
)

// AMH defines the AMH copula
type AMH struct{}

// Family returns the name of the copula family
func (c *AMH) Family() string {
	return "AMH"
}

// ThetaBounds returns the range where the copula is well defined
func (c *AMH) ThetaBounds() (float64, float64) {
	return 0., 1.
}

// Psi is the generating function of the copula
func (c *AMH) Psi(t float64, theta float64) float64 {
	return (1 - theta) / (math.Exp(t) - theta)
}

// PsiInv is the inverse of the generating function of the copula
func (c *AMH) PsiInv(t float64, theta float64) float64 {
	return math.Log(theta + (1-theta)/t)
}

// PsiD is the d-th derivative of Psi
func (c *AMH) PsiD(dim int, t float64, theta float64) float64 {
	coeff := 1.

	if dim%2 == 1 {
		coeff = -1.
	}

	// if theta == 0.0 {
	// 	return math.Exp(-t)
	// }
	// if theta == 1.0 {
	// 	return 0.0
	// }
	// return coeff * (1 - theta) * LiNeg(dim, theta*math.Exp(-t)) / theta
	return coeff * (1. - theta) * negativeIntegerPolylog(theta*math.Exp(-t), dim) / theta
}

// t computes  PsiInv(u_1) + PsiInv(u_2) ... + PsiInv(u_d)
func (c *AMH) t(vector []float64, theta float64) float64 {
	sum := 0.
	for _, x := range vector {
		sum += c.PsiInv(x, theta)
	}
	return sum
}

// Cdf computes the cumulative distribution function
// of the copula
func (c *AMH) Cdf(vector []float64, theta float64) float64 {
	return c.Psi(c.t(vector, theta), theta)
}

// Pdf computes the density of the generated copula
func (c *AMH) Pdf(vector []float64, theta float64) float64 {
	if theta == 0.0 {
		return 1.0
	}
	dim := len(vector)
	dimF := float64(dim)

	if min(vector) == 0 {
		p := math.Pow(1-theta, dimF+1)
		a := 1.0
		for _, uj := range vector {
			a = a * (1 - theta*(1-uj))
		}
		return p / (a * a)
	}

	a := 1.0
	h := theta
	for _, uj := range vector {
		vj := 1 - theta*(1-uj)
		a = a * uj * vj
		h = h * uj / vj
	}

	return math.Pow(1-theta, dimF+1) * negativeIntegerPolylog(h, dim) / (theta * a)
}

// LogPdf computes the logarithm of the
// density of the copula
func (c *AMH) LogPdf(vector []float64, theta float64) float64 {
	if min(vector) == 0. {
		return math.Inf(-1)
	}
	dim := len(vector)
	dimF := float64(dim)

	s1 := (dimF+1.)*math.Log(1.-theta) - 2*math.Log(theta)

	lh := 0.
	h := theta
	for j := range dim {
		h = h * vector[j] / (1. - theta*(1.-vector[j]))
		lh += math.Log(vector[j] * (1. - theta*(1.-vector[j])))
	}
	s2 := math.Log(theta) - lh

	s3 := math.Log(negativeIntegerPolylog(h, dim))

	return s1 + s2 + s3
}
