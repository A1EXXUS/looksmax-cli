package main

// RawMetrics mirrors one JSON line emitted by the Python CV worker
// (cv_worker/face_worker.py) on stdout. All fields except FaceDetected and
// Timestamp are heuristic geometric approximations, not medically or
// scientifically validated measurements -- see README.md for the
// disclaimer.
type RawMetrics struct {
	FaceDetected bool    `json:"face_detected"`
	Timestamp    float64 `json:"ts"`

	// Symmetry is a 0-100 score (100 = perfectly symmetric), derived from
	// comparing mirrored landmark pairs against the face's midline axis.
	Symmetry float64 `json:"symmetry"`

	// CanthalTiltDeg is the average eye-corner tilt in degrees; positive
	// means the outer corner sits above the inner corner.
	CanthalTiltDeg float64 `json:"canthal_tilt_deg"`

	// GonialAngleDeg approximates the mandible (jaw) angle in degrees.
	GonialAngleDeg float64 `json:"gonial_angle_deg"`

	// BigonialWidthRatio is jaw width relative to face height.
	BigonialWidthRatio float64 `json:"bigonial_width_ratio"`

	// CheekboneRatio is cheekbone width relative to overall face width.
	CheekboneRatio float64 `json:"cheekbone_ratio"`

	// FacialThirdsDev is the deviation (in %) of the three facial "thirds"
	// from perfect equality; 0 means the thirds are equal in length.
	FacialThirdsDev float64 `json:"facial_thirds_dev"`

	// The following three read best from a side profile (near frontal
	// they collapse toward ~180 deg, i.e. a near-straight line) -- same
	// caveat as Symmetry does the opposite way.

	// FacialConvexityDeg is the profile angle at the nose tip between the
	// glabella and the chin.
	FacialConvexityDeg float64 `json:"facial_convexity_deg"`

	// NasofrontalAngleDeg is the profile angle at the nasion between the
	// forehead and the nasal dorsum.
	NasofrontalAngleDeg float64 `json:"nasofrontal_angle_deg"`

	// NasolabialAngleDeg is the profile angle at the nose base between the
	// columella and the upper lip.
	NasolabialAngleDeg float64 `json:"nasolabial_angle_deg"`

	// FrontalityOffset is how far off-center the nose tip sits, as a
	// fraction of face width (~0 = frontal, grows toward profile). Used
	// by IsFrontalEnoughForScore to gate the composite score/tier.
	FrontalityOffset float64 `json:"frontality_offset"`

	// CheekboneProminence estimates cheekbone protrusion toward the camera
	// (via MediaPipe's per-landmark relative depth) rather than a flat
	// width ratio like CheekboneRatio. Experimental: monocular depth is
	// noisier and less validated than the plain x/y geometry above.
	CheekboneProminence float64 `json:"cheekbone_prominence"`
}
