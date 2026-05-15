package openai

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/api/handlers"
	log "github.com/sirupsen/logrus"
)

const (
	upscaleJobStatusQueued    = "queued"
	upscaleJobStatusRunning   = "running"
	upscaleJobStatusSucceeded = "succeeded"
	upscaleJobStatusFailed    = "failed"

	defaultUpscaleTargetLongEdge   = 4096
	defaultUpscaleWorkerStaleSec   = 900
	defaultUpscaleJobsRetentionSec = 7 * 24 * 60 * 60
)

type upscaleJob struct {
	ID             string         `json:"id"`
	Status         string         `json:"status"`
	SourceImageURL string         `json:"source_image_url"`
	ResultImageURL string         `json:"result_image_url"`
	SourceWidth    any            `json:"source_width"`
	SourceHeight   any            `json:"source_height"`
	OutputWidth    any            `json:"output_width"`
	OutputHeight   any            `json:"output_height"`
	TargetLongEdge int            `json:"target_long_edge"`
	Prompt         string         `json:"prompt"`
	Metadata       map[string]any `json:"metadata,omitempty"`
	GenerationSec  any            `json:"generation_sec"`
	UpscaleSec     any            `json:"upscale_sec"`
	ErrorMessage   string         `json:"error_message"`
	WorkerID       string         `json:"worker_id"`
	Attempts       int            `json:"attempts"`
	CreatedAt      string         `json:"created_at"`
	UpdatedAt      string         `json:"updated_at"`
	ClaimedAt      any            `json:"claimed_at"`
	HeartbeatAt    any            `json:"heartbeat_at"`
	FinishedAt     any            `json:"finished_at"`
	Summary        map[string]any `json:"summary,omitempty"`
}

type upscaleJobCreateRequest struct {
	SourceImageURL string         `json:"source_image_url"`
	SourceWidth    any            `json:"source_width"`
	SourceHeight   any            `json:"source_height"`
	TargetLongEdge int            `json:"target_long_edge"`
	Prompt         string         `json:"prompt"`
	Metadata       map[string]any `json:"metadata"`
	GenerationSec  any            `json:"generation_sec"`
}

type upscaleJobStore struct {
	mu          sync.Mutex
	path        string
	jobs        map[string]*upscaleJob
	order       []string
	workerStale time.Duration
	retention   time.Duration
	loaded      bool
}

var (
	upscaleStoreOnce sync.Once
	upscaleStore     *upscaleJobStore
)

func (h *OpenAIAPIHandler) CreateUpscaleJob(c *gin.Context) {
	var req upscaleJobCreateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		writeUpscaleError(c, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}
	job, err := getUpscaleJobStore().create(req)
	if err != nil {
		writeUpscaleError(c, http.StatusBadRequest, err.Error())
		return
	}
	c.JSON(http.StatusCreated, publicUpscaleJob(job))
}

func (h *OpenAIAPIHandler) GetUpscaleJob(c *gin.Context) {
	jobID := strings.TrimSpace(c.Param("job_id"))
	job, ok := getUpscaleJobStore().snapshot(jobID)
	if !ok {
		writeUpscaleError(c, http.StatusNotFound, "job not found")
		return
	}
	c.JSON(http.StatusOK, publicUpscaleJob(job))
}

func (h *OpenAIAPIHandler) ClaimUpscaleJob(c *gin.Context) {
	if !requireUpscaleWorkerAuth(c) {
		return
	}
	var body map[string]any
	_ = c.ShouldBindJSON(&body)
	workerID := strings.TrimSpace(valueAsString(body["worker_id"]))
	if workerID == "" {
		workerID = "unknown-worker"
	}
	job, ok, err := getUpscaleJobStore().claim(workerID)
	if err != nil {
		writeUpscaleError(c, http.StatusInternalServerError, err.Error())
		return
	}
	if !ok {
		c.Status(http.StatusNoContent)
		return
	}
	c.JSON(http.StatusOK, job)
}

func (h *OpenAIAPIHandler) HeartbeatUpscaleJob(c *gin.Context) {
	h.updateWorkerUpscaleJob(c, "heartbeat")
}

func (h *OpenAIAPIHandler) CompleteUpscaleJob(c *gin.Context) {
	h.updateWorkerUpscaleJob(c, "complete")
}

func (h *OpenAIAPIHandler) FailUpscaleJob(c *gin.Context) {
	h.updateWorkerUpscaleJob(c, "fail")
}

func (h *OpenAIAPIHandler) updateWorkerUpscaleJob(c *gin.Context, action string) {
	if !requireUpscaleWorkerAuth(c) {
		return
	}
	var body map[string]any
	if err := c.ShouldBindJSON(&body); err != nil {
		writeUpscaleError(c, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}
	jobID := strings.TrimSpace(c.Param("job_id"))
	workerID := strings.TrimSpace(valueAsString(body["worker_id"]))
	job, err := getUpscaleJobStore().workerUpdate(jobID, workerID, action, body)
	if err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, errUpscaleJobNotFound) {
			status = http.StatusNotFound
		} else if errors.Is(err, errUpscaleJobWorkerMismatch) {
			status = http.StatusConflict
		}
		writeUpscaleError(c, status, err.Error())
		return
	}
	c.JSON(http.StatusOK, publicUpscaleJob(job))
}

func getUpscaleJobStore() *upscaleJobStore {
	upscaleStoreOnce.Do(func() {
		upscaleStore = &upscaleJobStore{
			path:        upscaleJobsDataFile(),
			jobs:        make(map[string]*upscaleJob),
			workerStale: time.Duration(upscaleEnvInt("CPA_UPSCALE_WORKER_STALE_SECONDS", defaultUpscaleWorkerStaleSec, 1)) * time.Second,
			retention:   time.Duration(upscaleEnvInt("CPA_UPSCALE_JOBS_RETENTION_SECONDS", defaultUpscaleJobsRetentionSec, 60)) * time.Second,
		}
		if err := upscaleStore.load(); err != nil {
			// A corrupt queue file should not prevent CPA from serving normal API traffic.
			upscaleStore.jobs = make(map[string]*upscaleJob)
			upscaleStore.order = nil
		}
	})
	return upscaleStore
}

func (s *upscaleJobStore) create(req upscaleJobCreateRequest) (*upscaleJob, error) {
	sourceURL := strings.TrimSpace(req.SourceImageURL)
	if sourceURL == "" {
		return nil, errors.New("source_image_url is required")
	}
	target := req.TargetLongEdge
	if target <= 0 {
		target = defaultUpscaleTargetLongEdge
	}
	now := nowUpscaleISO()
	job := &upscaleJob{
		ID:             "up_" + strings.ReplaceAll(uuid.NewString(), "-", ""),
		Status:         upscaleJobStatusQueued,
		SourceImageURL: sourceURL,
		ResultImageURL: "",
		SourceWidth:    nullableNumber(req.SourceWidth),
		SourceHeight:   nullableNumber(req.SourceHeight),
		OutputWidth:    nil,
		OutputHeight:   nil,
		TargetLongEdge: target,
		Prompt:         strings.TrimSpace(req.Prompt),
		Metadata:       cloneMap(req.Metadata),
		GenerationSec:  nullableNumber(req.GenerationSec),
		UpscaleSec:     nil,
		ErrorMessage:   "",
		WorkerID:       "",
		Attempts:       0,
		CreatedAt:      now,
		UpdatedAt:      now,
		ClaimedAt:      nil,
		HeartbeatAt:    nil,
		FinishedAt:     nil,
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.withFileLockLocked(func() error {
		if err := s.loadLocked(); err != nil {
			return err
		}
		s.jobs[job.ID] = job
		s.order = append(s.order, job.ID)
		s.cleanupLocked(time.Now())
		return s.saveLocked()
	}); err != nil {
		return nil, err
	}
	return cloneUpscaleJob(job), nil
}

func (s *upscaleJobStore) snapshot(jobID string) (*upscaleJob, bool) {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return nil, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.withFileLockLocked(func() error {
		return s.loadLocked()
	}); err != nil {
		return nil, false
	}
	job, ok := s.jobs[jobID]
	if !ok || job == nil {
		return nil, false
	}
	return cloneUpscaleJob(job), true
}

func (s *upscaleJobStore) claim(workerID string) (*upscaleJob, bool, error) {
	now := time.Now()
	ts := now.UTC().Format(time.RFC3339)
	s.mu.Lock()
	defer s.mu.Unlock()
	var claimed *upscaleJob
	if err := s.withFileLockLocked(func() error {
		if err := s.loadLocked(); err != nil {
			return err
		}
		s.requeueStaleLocked(now)
		for _, id := range s.order {
			job := s.jobs[id]
			if job == nil || job.Status != upscaleJobStatusQueued {
				continue
			}
			job.Status = upscaleJobStatusRunning
			job.WorkerID = workerID
			job.ClaimedAt = ts
			job.HeartbeatAt = ts
			job.UpdatedAt = ts
			job.Attempts++
			job.ErrorMessage = ""
			if err := s.saveLocked(); err != nil {
				return err
			}
			claimed = cloneUpscaleJob(job)
			return nil
		}
		return nil
	}); err != nil {
		return nil, false, err
	}
	if claimed != nil {
		return claimed, true, nil
	}
	return nil, false, nil
}

var (
	errUpscaleJobNotFound       = errors.New("job not found")
	errUpscaleJobWorkerMismatch = errors.New("job is owned by another worker")
)

func (s *upscaleJobStore) workerUpdate(jobID string, workerID string, action string, body map[string]any) (*upscaleJob, error) {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return nil, errUpscaleJobNotFound
	}
	ts := nowUpscaleISO()
	s.mu.Lock()
	defer s.mu.Unlock()
	var updated *upscaleJob
	if err := s.withFileLockLocked(func() error {
		if err := s.loadLocked(); err != nil {
			return err
		}
		job := s.jobs[jobID]
		if job == nil {
			return errUpscaleJobNotFound
		}
		if job.WorkerID != "" && workerID != "" && job.WorkerID != workerID {
			return errUpscaleJobWorkerMismatch
		}
		switch action {
		case "heartbeat":
			job.HeartbeatAt = ts
			job.UpdatedAt = ts
		case "complete":
			job.Status = upscaleJobStatusSucceeded
			job.ResultImageURL = strings.TrimSpace(valueAsString(body["result_image_url"]))
			job.SourceWidth = coalesceNullableNumber(body["source_width"], job.SourceWidth)
			job.SourceHeight = coalesceNullableNumber(body["source_height"], job.SourceHeight)
			job.OutputWidth = nullableNumber(body["output_width"])
			job.OutputHeight = nullableNumber(body["output_height"])
			job.UpscaleSec = nullableNumber(body["upscale_sec"])
			job.Summary = cloneMapFromAny(body["summary"])
			job.ErrorMessage = ""
			job.HeartbeatAt = ts
			job.FinishedAt = ts
			job.UpdatedAt = ts
		case "fail":
			job.Status = upscaleJobStatusFailed
			job.ErrorMessage = strings.TrimSpace(valueAsString(body["error_message"]))
			if job.ErrorMessage == "" {
				job.ErrorMessage = "worker failed"
			}
			job.Summary = cloneMapFromAny(body["summary"])
			job.HeartbeatAt = ts
			job.FinishedAt = ts
			job.UpdatedAt = ts
		default:
			return fmt.Errorf("unsupported job action %q", action)
		}
		if err := s.saveLocked(); err != nil {
			return err
		}
		updated = cloneUpscaleJob(job)
		return nil
	}); err != nil {
		return nil, err
	}
	if action == "complete" || action == "fail" {
		cleanupUpscaleSourceObject(updated)
	}
	return updated, nil
}

func cleanupUpscaleSourceObject(job *upscaleJob) {
	key := upscaleSourceObjectKey(job)
	if key == "" || !deleteUpscaleSourceObjectEnabled() {
		return
	}
	storage, err := getImageObjectStorage()
	if err != nil {
		log.WithError(err).WithField("job_id", job.ID).Warn("upscale source cleanup skipped")
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := storage.deleteObject(ctx, key); err != nil {
		log.WithError(err).WithFields(log.Fields{
			"job_id": job.ID,
			"key":    key,
		}).Warn("upscale source cleanup failed")
		return
	}
	log.WithFields(log.Fields{
		"job_id": job.ID,
		"key":    key,
	}).Info("upscale source object deleted")
}

func upscaleSourceObjectKey(job *upscaleJob) string {
	if job == nil || len(job.Metadata) == 0 {
		return ""
	}
	return strings.Trim(strings.TrimSpace(valueAsString(job.Metadata["source_object_key"])), "/")
}

func deleteUpscaleSourceObjectEnabled() bool {
	raw := strings.TrimSpace(firstImageObjectEnv("CPA_UPSCALE_DELETE_SOURCE_OBJECT", "CPA_IMAGE_DELETE_UPSCALE_SOURCE"))
	if raw == "" {
		return true
	}
	return parseBoolField(raw, true)
}

func (s *upscaleJobStore) load() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.withFileLockLocked(func() error {
		return s.loadLocked()
	})
}

func (s *upscaleJobStore) loadLocked() error {
	s.loaded = true
	if strings.TrimSpace(s.path) == "" {
		return nil
	}
	data, err := os.ReadFile(s.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			s.jobs = make(map[string]*upscaleJob)
			s.order = nil
			return nil
		}
		return fmt.Errorf("upscale jobs: read store: %w", err)
	}
	var payload struct {
		Jobs []*upscaleJob `json:"jobs"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return fmt.Errorf("upscale jobs: parse store: %w", err)
	}
	s.jobs = make(map[string]*upscaleJob)
	s.order = nil
	for _, job := range payload.Jobs {
		if job == nil || strings.TrimSpace(job.ID) == "" {
			continue
		}
		normalizeLoadedUpscaleJob(job)
		s.jobs[job.ID] = job
		s.order = append(s.order, job.ID)
	}
	return nil
}

func normalizeLoadedUpscaleJob(job *upscaleJob) {
	job.SourceWidth = nullableNumber(job.SourceWidth)
	job.SourceHeight = nullableNumber(job.SourceHeight)
	job.OutputWidth = nullableNumber(job.OutputWidth)
	job.OutputHeight = nullableNumber(job.OutputHeight)
	job.GenerationSec = nullableNumber(job.GenerationSec)
	job.UpscaleSec = nullableNumber(job.UpscaleSec)
}

func (s *upscaleJobStore) withFileLockLocked(fn func() error) error {
	if strings.TrimSpace(s.path) == "" {
		return fn()
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return fmt.Errorf("upscale jobs: create store dir: %w", err)
	}
	lockPath := s.path + ".lock"
	lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("upscale jobs: open store lock: %w", err)
	}
	defer lockFile.Close()
	if err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX); err != nil {
		return fmt.Errorf("upscale jobs: lock store: %w", err)
	}
	defer func() {
		_ = syscall.Flock(int(lockFile.Fd()), syscall.LOCK_UN)
	}()
	return fn()
}

func (s *upscaleJobStore) saveLocked() error {
	if strings.TrimSpace(s.path) == "" {
		return nil
	}
	payload := struct {
		Jobs []*upscaleJob `json:"jobs"`
	}{Jobs: make([]*upscaleJob, 0, len(s.order))}
	for _, id := range s.order {
		if job := s.jobs[id]; job != nil {
			payload.Jobs = append(payload.Jobs, job)
		}
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("upscale jobs: marshal store: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return fmt.Errorf("upscale jobs: create store dir: %w", err)
	}
	tmpFile, err := os.CreateTemp(filepath.Dir(s.path), filepath.Base(s.path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("upscale jobs: create temp store: %w", err)
	}
	tmp := tmpFile.Name()
	if _, err := tmpFile.Write(data); err != nil {
		_ = tmpFile.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("upscale jobs: write temp store: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("upscale jobs: close temp store: %w", err)
	}
	if err := os.Chmod(tmp, 0o600); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("upscale jobs: chmod temp store: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("upscale jobs: replace store: %w", err)
	}
	return nil
}

func (s *upscaleJobStore) requeueStaleLocked(now time.Time) {
	if s.workerStale <= 0 {
		return
	}
	cutoff := now.Add(-s.workerStale)
	for _, job := range s.jobs {
		if job == nil || job.Status != upscaleJobStatusRunning {
			continue
		}
		seen := parseUpscaleTime(job.HeartbeatAt)
		if seen.IsZero() {
			seen = parseUpscaleTime(job.ClaimedAt)
		}
		if seen.IsZero() {
			seen = parseUpscaleTime(job.UpdatedAt)
		}
		if !seen.IsZero() && seen.Before(cutoff) {
			job.Status = upscaleJobStatusQueued
			job.WorkerID = ""
			job.UpdatedAt = now.UTC().Format(time.RFC3339)
			job.ErrorMessage = "requeued after stale worker heartbeat"
		}
	}
}

func (s *upscaleJobStore) cleanupLocked(now time.Time) {
	if s.retention <= 0 {
		return
	}
	cutoff := now.Add(-s.retention)
	nextOrder := s.order[:0]
	for _, id := range s.order {
		job := s.jobs[id]
		if job == nil {
			continue
		}
		if job.Status == upscaleJobStatusQueued || job.Status == upscaleJobStatusRunning {
			nextOrder = append(nextOrder, id)
			continue
		}
		ref := parseUpscaleTime(job.FinishedAt)
		if ref.IsZero() {
			ref = parseUpscaleTime(job.UpdatedAt)
		}
		if !ref.IsZero() && ref.Before(cutoff) {
			delete(s.jobs, id)
			continue
		}
		nextOrder = append(nextOrder, id)
	}
	s.order = nextOrder
}

func publicUpscaleJob(job *upscaleJob) gin.H {
	if job == nil {
		return nil
	}
	out := gin.H{
		"id":               job.ID,
		"status":           job.Status,
		"result_image_url": job.ResultImageURL,
		"source_width":     job.SourceWidth,
		"source_height":    job.SourceHeight,
		"output_width":     job.OutputWidth,
		"output_height":    job.OutputHeight,
		"target_long_edge": job.TargetLongEdge,
		"error_message":    job.ErrorMessage,
		"generation_sec":   job.GenerationSec,
		"upscale_sec":      job.UpscaleSec,
		"attempts":         job.Attempts,
		"created_at":       job.CreatedAt,
		"claimed_at":       job.ClaimedAt,
		"heartbeat_at":     job.HeartbeatAt,
		"finished_at":      job.FinishedAt,
		"updated_at":       job.UpdatedAt,
	}
	if metadata := publicUpscaleMetadata(job.Metadata); len(metadata) > 0 {
		out["metadata"] = metadata
	}
	if len(job.Summary) > 0 {
		out["summary"] = job.Summary
	}
	return out
}

func publicUpscaleMetadata(metadata map[string]any) map[string]any {
	if len(metadata) == 0 {
		return nil
	}
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		if key == "source_object_key" {
			continue
		}
		out[key] = value
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func requireUpscaleWorkerAuth(c *gin.Context) bool {
	secret := firstImageObjectEnv("CPA_UPSCALE_WORKER_SECRET", "WORKER_SECRET")
	if strings.TrimSpace(secret) == "" {
		writeUpscaleError(c, http.StatusServiceUnavailable, "upscale worker secret is not configured")
		return false
	}
	token := bearerToken(c.GetHeader("Authorization"))
	if subtle.ConstantTimeCompare([]byte(token), []byte(secret)) != 1 {
		writeUpscaleError(c, http.StatusUnauthorized, "unauthorized")
		return false
	}
	return true
}

func bearerToken(header string) string {
	header = strings.TrimSpace(header)
	if len(header) < 7 || !strings.EqualFold(header[:6], "Bearer") {
		return ""
	}
	return strings.TrimSpace(header[6:])
}

func writeUpscaleError(c *gin.Context, status int, message string) {
	if status <= 0 {
		status = http.StatusInternalServerError
	}
	if strings.TrimSpace(message) == "" {
		message = http.StatusText(status)
	}
	c.JSON(status, handlers.ErrorResponse{
		Error: handlers.ErrorDetail{
			Message: message,
			Type:    "invalid_request_error",
		},
	})
}

func upscaleJobsDataFile() string {
	for _, name := range []string{"CPA_UPSCALE_JOBS_FILE", "CPA_UPSCALE_JOB_DATA_FILE", "UPSCALE_JOBS_FILE", "DATA_FILE"} {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	return filepath.Join(os.TempDir(), "mino-cpa-upscale-jobs.json")
}

func upscaleEnvInt(name string, fallback int, min int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v < min {
		return fallback
	}
	return v
}

func nowUpscaleISO() string {
	return time.Now().UTC().Format(time.RFC3339)
}

func parseUpscaleTime(v any) time.Time {
	text := strings.TrimSpace(valueAsString(v))
	if text == "" {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339} {
		if ts, err := time.Parse(layout, text); err == nil {
			return ts
		}
	}
	return time.Time{}
}

func cloneUpscaleJob(job *upscaleJob) *upscaleJob {
	if job == nil {
		return nil
	}
	clone := *job
	clone.Metadata = cloneMap(job.Metadata)
	clone.Summary = cloneMap(job.Summary)
	return &clone
}

func cloneMap(in map[string]any) map[string]any {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneMapFromAny(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return cloneMap(m)
	}
	return nil
}

func valueAsString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case fmt.Stringer:
		return t.String()
	case nil:
		return ""
	default:
		return fmt.Sprintf("%v", t)
	}
}

func nullableNumber(v any) any {
	switch n := v.(type) {
	case nil:
		return nil
	case int:
		if n == 0 {
			return nil
		}
		return n
	case int64:
		if n == 0 {
			return nil
		}
		return normalizeNullableInteger(n)
	case float64:
		if n == 0 {
			return nil
		}
		if n == float64(int64(n)) {
			return normalizeNullableInteger(int64(n))
		}
		return n
	case json.Number:
		if i, err := n.Int64(); err == nil {
			if i == 0 {
				return nil
			}
			return normalizeNullableInteger(i)
		}
		if f, err := n.Float64(); err == nil {
			if f == 0 {
				return nil
			}
			return f
		}
		return nil
	default:
		text := strings.TrimSpace(valueAsString(v))
		if text == "" || text == "0" {
			return nil
		}
		if i, err := strconv.ParseInt(text, 10, 64); err == nil {
			if i == 0 {
				return nil
			}
			return normalizeNullableInteger(i)
		}
		if f, err := strconv.ParseFloat(text, 64); err == nil {
			if f == 0 {
				return nil
			}
			return f
		}
		return nil
	}
}

func normalizeNullableInteger(n int64) any {
	asInt := int(n)
	if int64(asInt) == n {
		return asInt
	}
	return n
}

func coalesceNullableNumber(v any, fallback any) any {
	if normalized := nullableNumber(v); normalized != nil {
		return normalized
	}
	return fallback
}
