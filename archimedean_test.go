// archimedean_test.go

package gopula

import (
	"fmt"
	"strings"
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
