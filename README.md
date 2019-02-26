# gopula

## Introduction

`gopula` is a pure Go package aimed to deal with Archimedean Copulas. It implements the well known families: Ali-Mikhail-Haq, Clayton, Frank, Gumbel and Joe. 

Currently `gopula` has two main features:
 - Parameter estimation through Maximum Likelihood (with confidence bounds)
 - Data sampling

To get it:

```shell
$ go get github.com/asiffer/gopula
```


## Get started
### Sampling

`gopula` uses the McNeil & Nešlehová universal sampling method to draw observations from a given archimedean copula (see [references](#references))

```go
// sampling.go

package main

import (
    "fmt"
    "gonum.org/v1/gonum/mat"
    "github.com/asiffer/gopula"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func main() {
    // Create a new instance of an Archimedean copula
    // NewCopula(family, theta)
    // Available families are:
    //  - "AMH"
    //  - "Clayton"
    //  - "Frank"
    //  - "Gumbel"
    //  - "Joe"
    A := gopula.NewCopula("Clayton", 2.5)

    // Sample some observations in the desired dimension
    // Sample(number of observations, dimension)
    // It returns a gonum matrix 
    M := A.Sample(9000, 3)
    matPrint(M)

    // The matrix can be exported to a csv file
    // SaveCSV(gonum matrix, path, separator)
    err := gopula.SaveCSV(M, "/tmp/data.csv", ',')
    if err != nil {
        fmt.Println(err)
    }
}
```

We can check the output file:
```shell
$ go run sampling.go
$ head /tmp/data.csv
0.517730,0.402611,0.704825
0.523264,0.565531,0.457385
0.110614,0.162048,0.117283
0.126623,0.175349,0.342992
0.205170,0.199397,0.390389
0.275976,0.151787,0.481059
0.149396,0.198625,0.085618
0.151417,0.586827,0.086791
0.143233,0.227281,0.126375
0.576612,0.459220,0.380677
$ wc -l /tmp/data.csv
9000 /tmp/data.csv
```

### Inference

Despite Archimedean copulas is quite a rich class of copulas with a great deal of nice properties, estimating the single parameter 𝜃 from observations is not so easy. Many techniques exist but `gopula` uses Maximum Likelihood Estimation as it performs rather the best (see the work of Marius Hofert, Martin Mächler and Alexander J. McNeil in [references](#references))

```go
// inference.go

package main

import (
    "fmt"
    "github.com/asiffer/gopula"
)

func main() {
    // Load the previous sampled dataset
    // Remember: they have been generated with 𝜃 = 2.5
    M, _ := gopula.LoadCSV("/tmp/data.csv", ',', false)
    if err != nil {
        fmt.Println(err)
        return
    }

    // Create a new instance of an Archimedean copula.
    // The initial value of theta does not matter in 
    // this case
    A := gopula.NewCopula("Clayton", 7.5)

    // Fit and see the results
    result := A.Fit(M)
    fmt.Println(result)
}
```

Let us check what the value of 𝜃 is infered:
```shell
$ go run inference.go
𝜃* 2.525        [2.481 2.570]
ℓ  10856.550
```

We can notice that 𝜃* is quite close to those we used to generate the data (the values in brackets are 95% confidence bounds and ℓ is the maximum likelihood reached).

## References

[[1]](https://projecteuclid.org/download/pdfview_1/euclid.aos/1247836677) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ1-norm symmetric distributions. The Annals of Statistics, 37(5B), 3059-3097.

[[2]](https://arxiv.org/pdf/1207.1708) Hofert, M., Mächler, M., & McNeil, A. J. (2012). Estimators for Archimedean copulas in high dimensions. arXiv preprint arXiv:1207.1708.
