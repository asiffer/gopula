// csv.go

package gopula

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func parseLine(line string, sep rune) ([]float64, error) {
	if sep == ' ' {
		// clean the multiple spaces
		for strings.Contains(line, "  ") {
			line = strings.Replace(line, "  ", " ", -1)
		}
	} else {
		// remove all the spaces
		line = strings.Replace(line, " ", "", -1)
	}
	line = strings.Replace(line, "\n", "", -1)

	splitLine := strings.Split(line, string(sep))

	values := make([]float64, len(splitLine))
	for i := 0; i < len(splitLine); i++ {
		v, err := strconv.ParseFloat(splitLine[i], 32)
		if err != nil {
			return values, err
		}
		values[i] = v
	}

	return values, nil
}

// LoadCSV loads data from a csvfile to a gonum matrix
func LoadCSV(path string, sep rune, header bool) (*mat.Dense, error) {
	var M *mat.Dense
	var line string
	nbRows := 0
	nbCols := -1

	rawVector := make([]float64, 0)

	f, err := os.Open(path)
	if err != nil {
		return M, err
	}

	defer f.Close()
	reader := bufio.NewReader(f)
	for err == nil {
		line, err = reader.ReadString('\n')
		if len(line) > 0 {
			values, errp := parseLine(line, sep)
			if errp != nil {
				return nil, errp
			}
			if nbCols == -1 {
				nbCols = len(values)
			}
			nbRows++ // a new valid row

			// we check if the dimension is correct
			if len(values) != nbCols {
				return nil, fmt.Errorf("The rows have not the same number of values (%d infered, go %d on line %d)", nbCols, len(values), nbRows)
			}

			rawVector = cat(rawVector, values)
		}

	}
	return mat.NewDense(nbRows, nbCols, rawVector), nil
}

// SaveCSV saves a gonum matrix to a csv file
func SaveCSV(M *mat.Dense, path string, sep rune) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}

	writer := bufio.NewWriter(f)
	n, _ := M.Dims()

	for i := 0; i < n; i++ {
		line := join(M.RawRowView(i), string(sep))
		_, err = writer.WriteString(line + "\n")
		if err != nil {
			return fmt.Errorf("[line %d] %s", i, err.Error())
		}
	}
	writer.Flush()
	f.Close()
	return nil
}
