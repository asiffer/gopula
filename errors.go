// errors.go

package gopula

import "errors"

var (
	// ErrUnknownFamily is returned when an unrecognised copula family name is given.
	ErrUnknownFamily = errors.New("unknown copula family")

	// ErrThetaOutOfBounds is returned when theta violates the mathematical
	// constraints of the requested copula family.
	ErrThetaOutOfBounds = errors.New("theta out of bounds for this family")

	// ErrInvalidInput is returned when copula inputs are not strictly in (0, 1).
	ErrInvalidInput = errors.New("copula inputs must be strictly in (0, 1)")

	// ErrSamplingFailed is returned when the radial PPF cannot converge for a
	// sample point after the maximum number of retries.
	ErrSamplingFailed = errors.New("radial PPF failed to converge for a sample point")
)
