// csv_test.go

package gopula

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"

	"gonum.org/v1/gonum/mat"
)

var testCSV = "resources/data.csv"

func getTheta(filename string) (float64, error) {
	filename = strings.ReplaceAll(filename, ".csv", "")
	i := strings.Index(filename, "_")
	if i == -1 {
		return 0., fmt.Errorf("invalid filename format: %s", filename)
	}
	thetaStr := strings.ReplaceAll(filename[i+1:], "_", ".")
	return strconv.ParseFloat(thetaStr, 64)
}

func loadCSVFromReader(r io.Reader, sep rune) (*mat.Dense, error) {
	reader := csv.NewReader(r)
	reader.Comma = sep
	reader.TrimLeadingSpace = true

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("empty CSV")
	}

	nbRows := len(records)
	nbCols := len(records[0])
	rawVector := make([]float64, 0, nbRows*nbCols)

	for i, row := range records { // no header
		if len(row) != nbCols {
			return nil, fmt.Errorf("row %d has %d columns, expected %d", i, len(row), nbCols)
		}
		for _, field := range row {
			v, err := strconv.ParseFloat(field, 64)
			if err != nil {
				return nil, err
			}
			rawVector = append(rawVector, v)
		}
	}
	return mat.NewDense(nbRows, nbCols, rawVector), nil
}

func loadCSV(path string, sep rune) (*mat.Dense, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return loadCSVFromReader(f, sep)
}

func loadCSVFromBytes(data []byte, sep rune) (*mat.Dense, error) {
	return loadCSVFromReader(bytes.NewReader(data), sep)
}

func saveCSV(M *mat.Dense, path string, sep rune) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	writer.Comma = sep

	n, p := M.Dims()
	for i := range n {
		row := make([]string, p)
		for j := range p {
			row[j] = strconv.FormatFloat(M.At(i, j), 'f', 10, 64)
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("[line %d] %s", i, err.Error())
		}
	}
	writer.Flush()
	return writer.Error()
}

func TestSaveCsv(t *testing.T) {
	n := 4
	p := 3
	M := mat.NewDense(n, p, nil)
	M.Set(0, 1, 3.5)
	err := saveCSV(M, testCSV, ',')
	if err != nil {
		t.Errorf("Error in saving matrix (%s)", err.Error())
	}
}

func TestLoadCsv(t *testing.T) {
	M, err := loadCSV(testCSV, ',')
	if err != nil {
		t.Fatal(err.Error())
	}
	n, p := M.Dims()
	if n != 4 || p != 3 {
		t.Errorf("Bad matrix dimensions (expected (4,3), got (%d, %d))", n, p)
	}
}
