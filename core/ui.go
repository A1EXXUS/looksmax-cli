package main

import (
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Disclaimer shown in the TUI footer on every frame. Also printed once to
// the terminal on startup (see main.go) so it can't be missed even if the
// user quits immediately.
const disclaimerText = "For fun only. Unvalidated, unscientific heuristics -- not a real assessment of anyone. See README."

// metricsMsg carries one parsed line from the Python worker's stdout.
type metricsMsg RawMetrics

// workerFatalMsg signals that the worker process died or could never start;
// stderr output (if any) is included for display.
type workerFatalMsg struct {
	err    error
	stderr string
}

// barSpec describes how to render one raw-metric progress bar.
type barSpec struct {
	label string
	value float64
	min   float64
	max   float64
	unit  string
}

// Model is the Bubble Tea application state for the live dashboard.
type Model struct {
	msgCh       chan tea.Msg
	weights     ScoreWeights
	metricsOnly bool
	noColor     bool

	faceDetected bool
	latest       RawMetrics
	lastUpdate   time.Time
	fatalErr     *workerFatalMsg
	quitting     bool
}

// NewModel builds the initial dashboard state.
func NewModel(msgCh chan tea.Msg, weights ScoreWeights, metricsOnly bool, noColor bool) Model {
	return Model{
		msgCh:       msgCh,
		weights:     weights,
		metricsOnly: metricsOnly,
		noColor:     noColor,
	}
}

// waitForActivity blocks on the shared message channel and forwards
// whatever the worker-reading goroutines send (metrics or a fatal error).
func waitForActivity(msgCh chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		return <-msgCh
	}
}

// Init starts the listener loop.
func (m Model) Init() tea.Cmd {
	return waitForActivity(m.msgCh)
}

// Update handles incoming worker data and key presses.
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			m.quitting = true
			return m, tea.Quit
		}

	case metricsMsg:
		m.faceDetected = msg.FaceDetected
		m.latest = RawMetrics(msg)
		m.lastUpdate = time.Now()
		return m, waitForActivity(m.msgCh)

	case workerFatalMsg:
		m.fatalErr = &msg
		return m, tea.Quit
	}

	return m, nil
}

// View renders the live dashboard.
func (m Model) View() string {
	if m.fatalErr != nil {
		return m.renderFatal()
	}

	var b strings.Builder
	b.WriteString(m.styleTitle().Render("looksmax-cli") + "\n\n")
	b.WriteString(m.renderStatus() + "\n\n")

	if m.faceDetected {
		b.WriteString(m.renderBars() + "\n")
		if !m.metricsOnly {
			b.WriteString("\n" + m.renderScore() + "\n")
		}
	}

	b.WriteString("\n" + m.renderFooter())
	return b.String()
}

func (m Model) renderFatal() string {
	style := m.style(lipgloss.Color("9")).Bold(true)
	var b strings.Builder
	b.WriteString(style.Render("looksmax-cli: worker stopped unexpectedly") + "\n\n")
	if m.fatalErr.err != nil {
		b.WriteString(m.fatalErr.err.Error() + "\n")
	}
	if strings.TrimSpace(m.fatalErr.stderr) != "" {
		b.WriteString(strings.TrimSpace(m.fatalErr.stderr) + "\n")
	}
	return b.String()
}

func (m Model) renderStatus() string {
	if !m.faceDetected {
		style := m.style(lipgloss.Color("3"))
		return style.Render("● no face detected") + "  (position your face in front of the camera)"
	}
	style := m.style(lipgloss.Color("2"))
	return style.Render("● face detected") + fmt.Sprintf("  (updated %s ago)", roundedSince(m.lastUpdate))
}

func roundedSince(t time.Time) string {
	if t.IsZero() {
		return "n/a"
	}
	d := time.Since(t).Round(100 * time.Millisecond)
	return d.String()
}

func (m Model) barSpecs() []barSpec {
	r := m.latest
	return []barSpec{
		{"symmetry", r.Symmetry, 0, 100, ""},
		{"canthal tilt", r.CanthalTiltDeg, -10, 20, "deg"},
		{"gonial angle", r.GonialAngleDeg, 90, 150, "deg"},
		{"bigonial width ratio", r.BigonialWidthRatio, 0.5, 1.1, ""},
		{"cheekbone ratio", r.CheekboneRatio, 0.5, 1.2, ""},
		{"facial thirds dev", r.FacialThirdsDev, 0, 40, "%"},
		{"facial convexity", r.FacialConvexityDeg, 140, 190, "deg"},
		{"nasofrontal angle", r.NasofrontalAngleDeg, 90, 190, "deg"},
		{"nasolabial angle", r.NasolabialAngleDeg, 70, 190, "deg"},
		// EXPERIMENTAL -- range picked from a single test session, not
		// real calibration data. See metrics.go's CheekboneProminence doc.
		{"cheek prominence", r.CheekboneProminence, -0.06, 0.02, ""},
	}
}

func (m Model) renderBars() string {
	var b strings.Builder
	labelWidth := 0
	for _, s := range m.barSpecs() {
		if len(s.label) > labelWidth {
			labelWidth = len(s.label)
		}
	}
	for _, s := range m.barSpecs() {
		b.WriteString(m.renderBar(s, labelWidth) + "\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

const barWidth = 24

func (m Model) renderBar(s barSpec, labelWidth int) string {
	frac := (s.value - s.min) / (s.max - s.min)
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	filled := int(frac * float64(barWidth))
	empty := barWidth - filled

	barStyle := m.style(lipgloss.Color("6"))
	bar := barStyle.Render(strings.Repeat("█", filled)) + strings.Repeat("░", empty)

	label := fmt.Sprintf("%-*s", labelWidth, s.label)
	valueStr := fmt.Sprintf("%.2f%s", s.value, s.unit)
	return fmt.Sprintf("%s  %s  %s", label, bar, valueStr)
}

func (m Model) renderScore() string {
	if !IsFrontalEnoughForScore(m.latest) {
		style := m.style(lipgloss.Color("3")).Italic(true)
		return style.Render("score: turn face frontal to rate (bilateral metrics need a near-frontal pose)")
	}

	score := CompositeScore(m.latest, m.weights)
	tier := TierForScore(score)

	tierColor := lipgloss.Color("6")
	switch tier {
	case TierSubhuman, TierLTN:
		tierColor = lipgloss.Color("9")
	case TierMTN:
		tierColor = lipgloss.Color("3")
	case TierHTN, TierChadlite:
		tierColor = lipgloss.Color("2")
	case TierChad:
		tierColor = lipgloss.Color("5")
	}

	scoreStyle := m.style(tierColor).Bold(true).Padding(0, 1)
	return scoreStyle.Render(fmt.Sprintf("score %.2f / %.1f  —  %s", score, scoreMax, string(tier)))
}

func (m Model) renderFooter() string {
	footerStyle := m.style(lipgloss.Color("8")).Italic(true)
	return footerStyle.Render(disclaimerText) + "\n" + footerStyle.Render("press q or ctrl+c to quit")
}

func (m Model) styleTitle() lipgloss.Style {
	return m.style(lipgloss.Color("5")).Bold(true)
}

// style returns a lipgloss style with the given foreground color, or a
// plain (colorless) style when --no-color was requested.
func (m Model) style(color lipgloss.Color) lipgloss.Style {
	if m.noColor {
		return lipgloss.NewStyle()
	}
	return lipgloss.NewStyle().Foreground(color)
}
