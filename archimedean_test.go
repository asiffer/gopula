// archimedean_test.go

package gopula

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

var (
	headerWidth = 80
)

func checkTitle(s string) {
	format := "%-" + fmt.Sprint(headerWidth-7) + "s"
	fmt.Printf(format, s)
}

func testOK() {
	fmt.Println("[\033[32mOK\033[0m]")
}

func testERROR() {
	fmt.Println("[\033[31mERROR\033[0m]")
}

func title(s string) {
	var l = len(s)
	var border int
	var left string
	var right string
	remaining := headerWidth - l - 2
	if remaining%2 == 0 {
		border = remaining / 2
		left = strings.Repeat("-", border) + " "
		right = " " + strings.Repeat("-", border)
	} else {
		border = (remaining - 1) / 2
		left = strings.Repeat("-", border+1) + " "
		right = " " + strings.Repeat("-", border)
	}

	fmt.Println(left + s + right)

}

func TestPrintFitResults(t *testing.T) {
	data, err := claytonFS.ReadFile("tests/samples/clayton_2.csv")
	if err != nil {
		t.Fatal(err)
	}
	M, err := loadCSVFromBytes(data, ',')
	if err != nil {
		t.Fatal(err)
	}

	AC, err := NewCopula("clayton", 1.0)
	if err != nil {
		t.Fatal(err)
	}
	result, _ := AC.Fit(M)
	fmt.Println(result)
}

func thetaRange(copuler ArchimedeanCopuler, n int) []float64 {
	min, max := copuler.ThetaBounds()
	if math.IsInf(max, 1) {
		max = inf
	}
	switch copuler.Family() {
	case "clayton", "joe":
		min += 1e-3
	case "amh":
		max -= 1e-3
	}
	p := make([]float64, n)
	step := (max - min) / float64(n-1)
	for i := 0; i < n; i++ {
		p[i] = min + step*float64(i)
	}
	return p
}

func TestRadialCdf(t *testing.T) {
	title("Testing radial cdf")

	dim := 2
	data := map[string][][]float64{
		"amh":     AMH_RADIAL_CDF,
		"clayton": CLAYTON_RADIAL_CDF,
		"gumbel":  GUMBEL_RADIAL_CDF,
		"frank":   FRANK_RADIAL_CDF,
		"joe":     JOE_RADIAL_CDF,
	}

	for family, rows := range data {
		t.Run(family, func(st *testing.T) {
			for _, row := range rows {
				theta := row[0]
				r := row[1]
				expected := row[2]
				c, err := NewCopula(family, theta)
				if err != nil {
					st.Fatal(err)
				}
				computed := c.RadialCdf(r, dim)
				if math.Abs(expected-computed) > 1e-8 {
					st.Errorf("Bad radial computation for theta=%.2f, r=%.2f, dim=%d: expected %.5f, got %.5f",
						theta, r, dim, expected, computed,
					)
				}
			}
		})
	}

}
