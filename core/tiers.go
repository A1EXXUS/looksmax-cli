package main

import (
	"fmt"
	"math"
	"os"

	"gopkg.in/yaml.v3"
)

// --- Calibration constants -------------------------------------------------
//
// Everything below (ideal targets, tolerance/"sigma" values, and weights) is
// an arbitrary, hand-picked heuristic loosely inspired by informal
// "looksmaxxing" community lore. None of it is derived from any scientific
// or medical study. It exists purely to turn a handful of face-geometry
// numbers into one entertaining composite score -- treat the exact values
// as a subjective, adjustable opinion, not ground truth.

const (
	idealCanthalTiltDeg  = 4.0
	canthalTiltSigma     = 6.0
	idealGonialAngleDeg  = 120.0
	gonialAngleSigma     = 15.0
	idealBigonialWidth   = 0.78
	bigonialWidthSigma   = 0.12
	idealCheekboneRatio  = 0.88
	cheekboneRatioSigma  = 0.15
	facialThirdsDevSigma = 8.0

	// Composite score range, loosely modeled after the informal PSL
	// ("Personal Sexual/Self-rated Looks") scale used in some online
	// communities. 1.0 = lowest, 8.5 = highest.
	scoreMin = 1.0
	scoreMax = 8.5

	// Above this, the head is turned far enough that the bilateral-pair
	// metrics (symmetry, gonial angle, bigonial width, cheekbone ratio)
	// can no longer be trusted, so the composite score/tier is hidden
	// rather than shown wrong. Keep in sync with cv_worker/face_worker.py's
	// _FRONTALITY_MAX.
	maxFrontalityOffsetForScore = 0.22
)

// Weights default to summing to 1.0. They can be overridden via an optional
// YAML config file (see LoadScoreConfig); if the overrides don't sum to 1.0
// they are re-normalized rather than rejected.
const (
	defaultWeightSymmetry     = 0.25
	defaultWeightCanthalTilt  = 0.15
	defaultWeightGonialAngle  = 0.15
	defaultWeightBigonialW    = 0.15
	defaultWeightCheekbone    = 0.15
	defaultWeightFacialThirds = 0.15
)

// Tier is one of the composite-score bands used for the final verdict.
type Tier string

const (
	TierSubhuman Tier = "Subhuman"
	TierLTN      Tier = "LTN"
	TierMTN      Tier = "MTN"
	TierHTN      Tier = "HTN"
	TierChadlite Tier = "Chadlite"
	TierChad     Tier = "Chad"
)

// ScoreWeights holds the relative contribution of each raw metric to the
// final composite score.
type ScoreWeights struct {
	Symmetry     float64 `yaml:"symmetry"`
	CanthalTilt  float64 `yaml:"canthal_tilt"`
	GonialAngle  float64 `yaml:"gonial_angle"`
	BigonialW    float64 `yaml:"bigonial_width"`
	Cheekbone    float64 `yaml:"cheekbone"`
	FacialThirds float64 `yaml:"facial_thirds"`
}

// DefaultScoreWeights returns the built-in weight calibration.
func DefaultScoreWeights() ScoreWeights {
	return ScoreWeights{
		Symmetry:     defaultWeightSymmetry,
		CanthalTilt:  defaultWeightCanthalTilt,
		GonialAngle:  defaultWeightGonialAngle,
		BigonialW:    defaultWeightBigonialW,
		Cheekbone:    defaultWeightCheekbone,
		FacialThirds: defaultWeightFacialThirds,
	}
}

// scoreConfigFile is the on-disk shape of an optional --config YAML file.
type scoreConfigFile struct {
	Weights ScoreWeights `yaml:"weights"`
}

// LoadScoreWeights reads weight overrides from a YAML file at path. An empty
// path returns the built-in defaults untouched. Weights that don't sum to
// 1.0 are normalized rather than treated as an error, since this is a
// for-fun scoring knob, not a strict contract.
func LoadScoreWeights(path string) (ScoreWeights, error) {
	weights := DefaultScoreWeights()
	if path == "" {
		return weights, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return weights, fmt.Errorf("reading score config %q: %w", path, err)
	}

	var cfg scoreConfigFile
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return weights, fmt.Errorf("parsing score config %q: %w", path, err)
	}

	overridden := cfg.Weights
	sum := overridden.Symmetry + overridden.CanthalTilt + overridden.GonialAngle +
		overridden.BigonialW + overridden.Cheekbone + overridden.FacialThirds
	if sum <= 0 {
		// Nothing usable in the file (e.g. empty/malformed weights block);
		// keep the defaults instead of dividing by zero.
		return weights, nil
	}

	return ScoreWeights{
		Symmetry:     overridden.Symmetry / sum,
		CanthalTilt:  overridden.CanthalTilt / sum,
		GonialAngle:  overridden.GonialAngle / sum,
		BigonialW:    overridden.BigonialW / sum,
		Cheekbone:    overridden.Cheekbone / sum,
		FacialThirds: overridden.FacialThirds / sum,
	}, nil
}

// gaussianSubscore scores how close value is to ideal on a 0-100 scale,
// falling off smoothly as value drifts away from ideal by more than sigma.
func gaussianSubscore(value, ideal, sigma float64) float64 {
	if sigma <= 0 {
		return 0
	}
	diff := value - ideal
	return 100.0 * math.Exp(-(diff*diff)/(2*sigma*sigma))
}

// CompositeScore turns a set of raw metrics into a single 1.0-8.5 score
// using the weighted, heuristic formula described above. Metrics with a
// FaceDetected == false value should not be passed in.
func CompositeScore(m RawMetrics, weights ScoreWeights) float64 {
	symmetrySub := math.Max(0, math.Min(100, m.Symmetry))
	canthalSub := gaussianSubscore(m.CanthalTiltDeg, idealCanthalTiltDeg, canthalTiltSigma)
	gonialSub := gaussianSubscore(m.GonialAngleDeg, idealGonialAngleDeg, gonialAngleSigma)
	bigonialSub := gaussianSubscore(m.BigonialWidthRatio, idealBigonialWidth, bigonialWidthSigma)
	cheekboneSub := gaussianSubscore(m.CheekboneRatio, idealCheekboneRatio, cheekboneRatioSigma)
	thirdsSub := gaussianSubscore(m.FacialThirdsDev, 0, facialThirdsDevSigma)

	composite := weights.Symmetry*symmetrySub +
		weights.CanthalTilt*canthalSub +
		weights.GonialAngle*gonialSub +
		weights.BigonialW*bigonialSub +
		weights.Cheekbone*cheekboneSub +
		weights.FacialThirds*thirdsSub

	composite = math.Max(0, math.Min(100, composite))
	return scoreMin + (composite/100.0)*(scoreMax-scoreMin)
}

// IsFrontalEnoughForScore reports whether the current frame's head pose is
// close enough to frontal that the composite score/tier can be trusted.
func IsFrontalEnoughForScore(m RawMetrics) bool {
	return m.FrontalityOffset <= maxFrontalityOffsetForScore
}

// TierForScore maps a composite 1.0-8.5 score to its named tier, per the
// informal PSL-style banding described in the project spec.
func TierForScore(score float64) Tier {
	switch {
	case score < 3.5:
		return TierSubhuman
	case score < 5.0:
		return TierLTN
	case score < 6.0:
		return TierMTN
	case score < 7.0:
		return TierHTN
	case score < 8.0:
		return TierChadlite
	default:
		return TierChad
	}
}
