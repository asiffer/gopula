// utils.go

package gopula

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func zeros(n int) []float64 {
	return make([]float64, n)
}

func ones(n int) []float64 {
	v := zeros(n)
	for i := 0; i < n; i++ {
		v[i] = 1.
	}
	return v
}

func arange(start float64, end float64, step float64) []float64 {
	length := int(math.Floor((end - start) / step))
	if length > 0 {
		v := zeros(length)
		v[0] = start
		for i := 1; i < length; i++ {
			v[i] = v[i-1] + step
		}
		return v
	}
	return nil
}

func min(v []float64) float64 {
	size := len(v)
	min := 0.
	if size > 0 {
		min = v[0]
		for i := 1; i < size; i++ {
			if v[i] < min {
				min = v[i]
			}
		}
	}
	return min
}

func max(v []float64) float64 {
	size := len(v)
	max := 0.
	if size > 0 {
		max = v[0]
		for i := 1; i < size; i++ {
			if v[i] > max {
				max = v[i]
			}
		}
	}
	return max
}

func sum(v []float64) float64 {
	s := 0.
	for _, x := range v {
		s += x
	}
	return s
}

func prod(v []float64) float64 {
	p := 1.
	for _, x := range v {
		p = p * x
	}
	return p
}

func createCopy(v []float64) []float64 {
	output := make([]float64, len(v))
	copy(output, v)
	return output
}

func cat(base []float64, add []float64) []float64 {
	return append(base, add...)
}

func scalarMul(v []float64, lambda float64) []float64 {

	output := createCopy(v)
	for i := 0; i < len(v); i++ {
		output[i] = output[i] * lambda
	}
	return output
}

func scalarAdd(v []float64, c float64) []float64 {
	output := createCopy(v)
	for i := 0; i < len(v); i++ {
		output[i] = v[i] + c
	}
	return output
}

func scalarDiv(v []float64, lambda float64) []float64 {
	return scalarMul(v, 1./lambda)
}

func scalarSub(v []float64, c float64) []float64 {
	return scalarAdd(v, -c)
}

func pow(v []float64, e float64) []float64 {
	output := createCopy(v)
	for i := 0; i < len(v); i++ {
		output[i] = math.Pow(v[i], e)
	}
	return output
}

func log(v []float64) []float64 {
	output := createCopy(v)
	for i := 0; i < len(v); i++ {
		output[i] = math.Log(v[i])
	}
	return output
}

func join(v []float64, sep string) string {
	p := len(v)
	format := "%f" + strings.Repeat(sep+"%f", p-1)
	iface := make([]interface{}, p)
	for i := 0; i < p; i++ {
		iface[i] = v[i]
	}
	return fmt.Sprintf(format, iface...)
}

func factorial(n int) int {
	if n == 0 {
		return 1
	}
	return n * factorial(n-1)
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func uniformSample(n int) []float64 {
	sample := make([]float64, n)
	for i := 0; i < n; i++ {
		sample[i] = rand.Float64()
	}
	// fmt.Println(sample)
	return sample
}

func standardExpSample(n int) []float64 {
	sample := make([]float64, n)
	for i := 0; i < n; i++ {
		sample[i] = rand.ExpFloat64()
	}
	return sample
}

func euclid(a int, b int) (int, int) {
	if a >= 0 && b > 0 {
		r := a % b
		q := (a - r) / b
		return q, r
	}
	return 0, 0
}
