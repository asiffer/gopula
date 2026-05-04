# gopula [![Build Status](https://travis-ci.com/asiffer/gopula.svg?branch=master)](https://travis-ci.com/asiffer/gopula) [![Go Report Card](https://goreportcard.com/badge/github.com/asiffer/gopula)](https://goreportcard.com/report/github.com/asiffer/gopula) [![Coverage Status](https://codecov.io/github/asiffer/gopula/coverage.svg?branch=master)](https://codecov.io/github/asiffer/gopula?branch=master) [![GoDoc](https://godoc.org/github.com/asiffer/gopula?status.svg)](https://godoc.org/github.com/asiffer/gopula) 


`gopula` is a pure Go package aimed to deal with Archimedean Copulas. 

It implements the well known families: Ali-Mikhail-Haq, Clayton, Frank, Gumbel and Joe.

Currently `gopula` has three main features:
 - Basic computations (pdf, cdf, etc.)
 - Parameter estimation through Maximum Likelihood (with confidence bounds)
 - Data sampling

To get it:

```shell
$ go get github.com/asiffer/gopula
```


## Get started

```go
// main.go
package main

import (
    "fmt"

    "gonum.org/v1/gonum/mat"
    "github.com/asiffer/gopula"
)

func main() {
    // Create a new instance of an Archimedean copula
    // NewCopula(family, theta)
    // Available families are:
    //  - "AMH"
    //  - "Clayton"
    //  - "Frank"
    //  - "Gumbel"
    //  - "Joe"
    // it triggers an error if theta is not in the right
    // bounds of the family (or if the family is unknown)
    A, err := gopula.NewCopula("Clayton", 2.5)
    if err != nil {
        panic(err)
    }

    // Sample some observations in the desired dimension
    // Sample(number of observations, dimension)
    // It returns a gonum matrix 
    // It can returns an error when the radial PPF cannot converge for some point
    M, err := A.Sample(10000, 3)
    
    // let us modify theta
    A.theta = 10.0
    // fit from the sample and show results
    // an error can trigger we the minimum of the 
    // log-likelihood cannot be found
    result, err := A.Fit(M)
    fmt.Println(result)
}
```

We can notice that 𝜃* is quite close to those we used to generate the data (the values in brackets are 95% confidence bounds and ℓ is the maximum likelihood reached).

```shell
       ℓ 12171.392848
       𝜃 2.489516
     95% [2.448, 2.532]
   Evals 18
 Message Success
```

## Details

### Sampling

`gopula` uses the McNeil & Nešlehová universal sampling method to draw observations from a given archimedean copula (see [references](#references))

### Inference

Despite Archimedean copulas is quite a rich class of copulas with a great deal of nice properties, estimating the single parameter 𝜃 from observations is not so easy. Many techniques exist but `gopula` uses **Maximum Likelihood Estimation** (MLE) as it performs rather the best (see the work of Marius Hofert, Martin Mächler and Alexander J. McNeil in [[2]](#references)).

The inference procedure mainly uses the formulas provided by the authors mentionned above (see [[3]](#references))


## References

[[1]](https://projecteuclid.org/download/pdfview_1/euclid.aos/1247836677) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ1-norm symmetric distributions. The Annals of Statistics, 37(5B), 3059-3097.

[[2]](https://arxiv.org/pdf/1207.1708) Hofert, M., Mächler, M., & McNeil, A. J. (2012). Estimators for Archimedean copulas in high dimensions. arXiv preprint arXiv:1207.1708.

[[3]](https://www.sciencedirect.com/science/article/pii/S0047259X12000607) Hofert, M., Mächler, M., & Mcneil, A. J. (2012). Likelihood inference for Archimedean copulas in high dimensions under known margins. Journal of Multivariate Analysis, 110, 133-150.