// joe.go

package gopula

import (
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
	return 1., math.Inf(1)
}

// Psi is the generating function of the copula
func (c *Joe) Psi(t float64, theta float64) float64 {
	// return 1. - math.Pow(1.-math.Exp(-t), 1./theta)
	return 1.0 - math.Pow(-math.Expm1(-t), 1.0/theta)
}

// PsiInv is the inverse of the generating function of the copula
func (c *Joe) PsiInv(t float64, theta float64) float64 {
	// return -math.Log(1 - math.Pow(1.-t, theta))
	return -math.Log1p(-math.Pow(1.-t, theta))
}

func joeCoeff(dim int, k int, alpha float64) float64 {
	return stirling2(dim, k+1) * math.Gamma(float64(k+1)-alpha) / math.Gamma(1.-alpha)
}

func joePolynom(x float64, dim int, alpha float64) float64 {
	if x == 0.0 {
		return 1.0
	}
	p := 0.
	xk := 1. // x^0
	for k := range dim {
		p += joeCoeff(dim, k, alpha) * xk
		xk = xk * x
	}
	return p
}

func joeH(vector []float64, dim int, theta float64) float64 {
	h := 1.
	for j := range dim {
		h = h * (1 - math.Pow(1-vector[j], theta))
	}
	return h
}

// PsiD is the d-th derivative of Psi
// func (c *Joe) PsiD(d int, t float64, theta float64) float64 {
// 	coeff := 1.
// 	alpha := 1. / theta
// 	if d%2 == 1 {
// 		coeff = -1.
// 	}
// 	e := math.Exp(-t)
// 	return coeff * alpha * e * joePolynom(e/(1.-e), d, alpha) / math.Pow(1.-e, 1.-alpha)
// }

func (c *Joe) PsiD(d int, t float64, theta float64) float64 {
	coeff := 1.0
	alpha := 1.0 / theta
	if d%2 == 1 {
		coeff = -1.0
	}
	me := -math.Expm1(-t) // 1 - math.Exp(-t)
	e := 1 - me           // math.Exp(-t)

	return coeff * alpha * (e / math.Pow(me, 1.0-alpha)) * joePolynom(e/me, d, alpha)
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
	// if min(vector) == 0. {
	// 	return 0.
	// }
	// return math.Exp(c.LogPdf(vector, theta))

	dim := len(vector)
	dimF := float64(dim)
	alpha := 1. / theta

	// if min(vector) >= 0.95 {
	// 	epsilon := 1 - mean(vector)
	// 	return math.Pow(theta, dimF-1.) *
	// 		math.Pow(dimF, alpha-dimF) *
	// 		math.Pow(epsilon, 1-dimF)
	// }

	h := 1.
	num := 1.
	for j := range dim {
		h = h * (1 - math.Pow(1-vector[j], theta))
		// num = num * math.Pow(1-vector[j], theta-1)
		num = num * (1 - vector[j])
	}

	num = math.Pow(num, theta-1.0)

	if min(vector) == 0.0 {
		return math.Pow(theta, dimF-1.) * num
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

	sumLog1mU := 0.
	logH := 0.
	for j := range dim {
		sumLog1mU += math.Log1p(-vector[j])
		logH += math.Log1p(-math.Pow(1-vector[j], theta))
	}

	// log(1-h) via -expm1(log h) stays finite even when h rounds to 1.
	log1mH := math.Log(-math.Expm1(logH))

	// T = Σ_k a_{dk}(α) h^k (1-h)^(d-1-k); equals (1-h)^(d-1) · P^J(h/(1-h)),
	// so log P^J(h/(1-h)) = log T - (d-1) log(1-h). The sum has only
	// non-negative terms and stays well-conditioned at both h≈0 and h≈1.
	T := 0.
	for k := 0; k < dim; k++ {
		T += joeCoeff(dim, k, alpha) *
			math.Exp(float64(k)*logH+float64(dim-1-k)*log1mH)
	}

	return (dimF-1)*math.Log(theta) +
		(theta-1)*sumLog1mU +
		math.Log(T) -
		(dimF-alpha)*log1mH
}
