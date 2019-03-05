// frank.go

package gopula

import (
	"math"
)

// Frank defines the Frank copula
type Frank struct{}

// Family returns the name of the copula family
func (c *Frank) Family() string {
	return "Frank"
}

// ThetaBounds returns the range where the copula is well defined
func (c *Frank) ThetaBounds() (float64, float64) {
	return 0., Inf
}

// Psi is the generating function of the copula
func (c *Frank) Psi(t float64, theta float64) float64 {
	return -math.Log(1.-(1.-math.Exp(-theta))*math.Exp(-t)) / theta
}

// PsiInv is the inverse of the generating function of the copula
func (c *Frank) PsiInv(t float64, theta float64) float64 {
	return -math.Log((1. - math.Exp(-theta*t)) / (1. - math.Exp(-theta)))
}

func negativeIntegerPolylog(x float64, dim int) float64 {
	// Woods formula to compute Li_{-d}(x)
	Li := 0.
	for k := 0; k < dim+1; k++ {
		Li += float64(factorial(k)) * stirlingSecondKind.At(dim+1, k+1) * math.Pow(x/(1.-x), float64(k+1))
	}
	return Li
}

// PsiD is the d-th derivative of Psi
func (c *Frank) PsiD(dim int, t float64, theta float64) float64 {
	coeff := 1.
	alpha := 1. / theta
	if dim%2 == 1 {
		coeff = -1.
	}
	return coeff * alpha * negativeIntegerPolylog((1.-math.Exp(-theta))*math.Exp(-t), dim-1)
}

// t computes  PsiInv(u_1) + PsiInv(u_2) ... + PsiInv(u_d)
func (c *Frank) t(vector []float64, theta float64) float64 {
	sum := 0.
	for _, x := range vector {
		sum += c.PsiInv(x, theta)
	}
	return sum
}

// Cdf computes the cumulative distribution function
// of the copula
func (c *Frank) Cdf(vector []float64, theta float64) float64 {
	return c.Psi(c.t(vector, theta), theta)
}

// Pdf computes the density of the generated copula
func (c *Frank) Pdf(vector []float64, theta float64) float64 {
	if min(vector) == 0. {
		return 0.
	}
	dim := len(vector)
	dimF := float64(dim)

	r := 1. - math.Exp(-theta)
	p := math.Pow(theta/r, dimF-1.)
	h := math.Pow(r, 1.-dimF)
	for j := 0; j < dim; j++ {
		h = h * (1. - math.Exp(-theta*vector[j]))
	}

	return p * negativeIntegerPolylog(h, dim-1) * math.Exp(-theta*sum(vector)) / h
}

// LogPdf computes the logarithm of the
// density of the copula
func (c *Frank) LogPdf(vector []float64, theta float64) float64 {
	if min(vector) == 0. {
		return math.Inf(-1)
	}
	dim := len(vector)
	dimF := float64(dim)

	r := 1. - math.Exp(-theta)
	h := math.Pow(r, 1.-dimF)
	lh := 0.
	for j := 0; j < dim; j++ {
		w := (1. - math.Exp(-theta*vector[j]))
		h = h * w
		lh += math.Log(w)
	}

	s1 := (dimF - 1.) * math.Log(theta/r)
	s2 := math.Log(negativeIntegerPolylog(h, dim-1))
	s3 := theta * sum(vector)
	s4 := (1.-dimF)*math.Log(r) + lh
	return s1 + s2 - s3 - s4
}
