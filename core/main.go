// Command looksmax-cli is a terminal dashboard that streams live,
// heuristic facial-geometry metrics from a webcam. See README.md for the
// full disclaimer: this is a for-fun portfolio project, not a scientific or
// medical tool.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	pythonPath := flag.String("python-path", "python3", "path to the Python interpreter used to run the CV worker")
	cameraIndex := flag.Int("camera-index", 0, "OpenCV camera device index to capture from")
	noColor := flag.Bool("no-color", false, "disable colored output")
	metricsOnly := flag.Bool("metrics-only", false, "show only raw metrics, without the composite score/tier")
	configPath := flag.String("config", "", "optional YAML file overriding the composite-score weights")
	flag.Parse()

	fmt.Println(disclaimerText)
	fmt.Println()

	weights, err := LoadScoreWeights(*configPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error: "+err.Error())
		os.Exit(1)
	}

	workerScript, err := resolveWorkerScript()
	if err != nil {
		fmt.Fprintln(os.Stderr, "error: "+err.Error())
		os.Exit(1)
	}

	if _, err := exec.LookPath(*pythonPath); err != nil {
		fmt.Fprintf(os.Stderr, "error: could not find Python interpreter %q (use --python-path to point at one)\n", *pythonPath)
		os.Exit(1)
	}

	cmd := exec.Command(*pythonPath, workerScript, "--camera-index", strconv.Itoa(*cameraIndex))

	// The worker watches its stdin for EOF as a signal that we've died and
	// it should shut down (see cv_worker/face_worker.py's StdinWatcher). If
	// we leave cmd.Stdin nil, Go connects the child to /dev/null, which
	// reads as EOF instantly and makes the worker exit before it ever
	// streams a frame. Give it a pipe we control instead, and only close it
	// (in killWorker) when we actually want it to shut down.
	stdinReader, stdinWriter := io.Pipe()
	cmd.Stdin = stdinReader

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fmt.Fprintln(os.Stderr, "error: could not attach to worker stdout: "+err.Error())
		os.Exit(1)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		fmt.Fprintln(os.Stderr, "error: could not attach to worker stderr: "+err.Error())
		os.Exit(1)
	}

	if err := cmd.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "error: could not start CV worker (%s %s): %s\n", *pythonPath, workerScript, err.Error())
		os.Exit(1)
	}

	msgCh := make(chan tea.Msg, 16)
	stderrBuf := &strings.Builder{}

	// Forward each JSON line from the worker's stdout as a metricsMsg.
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.TrimSpace(line) == "" {
				continue
			}
			var raw RawMetrics
			if err := json.Unmarshal([]byte(line), &raw); err != nil {
				continue // ignore malformed lines rather than crashing the UI
			}
			msgCh <- metricsMsg(raw)
		}
	}()

	// Capture stderr so a fatal worker error (e.g. no camera) can be shown.
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			stderrBuf.WriteString(scanner.Text() + "\n")
		}
	}()

	// When the worker process exits, surface that to the UI if it wasn't a
	// clean shutdown we initiated ourselves.
	workerDone := make(chan error, 1)
	go func() {
		workerDone <- cmd.Wait()
	}()

	model := NewModel(msgCh, weights, *metricsOnly, *noColor)
	program := tea.NewProgram(model)

	shuttingDown := make(chan struct{})
	go func() {
		select {
		case err := <-workerDone:
			select {
			case <-shuttingDown:
				return // we killed it ourselves on quit; nothing to report
			default:
			}
			msgCh <- workerFatalMsg{err: err, stderr: stderrBuf.String()}
		case <-shuttingDown:
			return
		}
	}()

	// Forward OS interrupt/terminate signals into a graceful shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		program.Quit()
	}()

	_, runErr := program.Run()

	close(shuttingDown)
	killWorker(cmd, stdinWriter)

	if runErr != nil {
		fmt.Fprintln(os.Stderr, "error: "+runErr.Error())
		os.Exit(1)
	}
}

// killWorker asks the CV worker to shut down gracefully -- closing its
// stdin (the worker's own preferred shutdown signal) and sending SIGTERM as
// a backstop -- giving it a short grace period before forcing termination.
func killWorker(cmd *exec.Cmd, stdinWriter io.Closer) {
	if cmd.Process == nil {
		return
	}

	done := make(chan struct{})
	go func() {
		_ = cmd.Wait()
		close(done)
	}()

	_ = stdinWriter.Close()
	_ = cmd.Process.Signal(syscall.SIGTERM)

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		_ = cmd.Process.Kill()
	}
}

// resolveWorkerScript locates cv_worker/face_worker.py relative to the
// repository layout (core/ and cv_worker/ are sibling directories). It
// tries a few likely locations so the binary works whether it's run via
// `go run ./...` from core/, a built binary in core/, or from the repo root.
func resolveWorkerScript() (string, error) {
	candidates := []string{
		filepath.Join("..", "cv_worker", "face_worker.py"),
		filepath.Join("cv_worker", "face_worker.py"),
	}

	if exePath, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exePath)
		candidates = append(candidates,
			filepath.Join(exeDir, "..", "cv_worker", "face_worker.py"),
			filepath.Join(exeDir, "cv_worker", "face_worker.py"),
		)
	}

	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("could not locate cv_worker/face_worker.py (looked in: %s)", strings.Join(candidates, ", "))
}
