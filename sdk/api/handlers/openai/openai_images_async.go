package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	internalconfig "github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/api/handlers"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	defaultImageAsyncWorkers        = 2000
	defaultImageAsyncQueueSize      = 20000
	defaultImageAsyncTimeoutSeconds = 1800
	defaultImageAsyncTTLSeconds     = 86400
)

const (
	imageTaskStatusQueued    = "queued"
	imageTaskStatusRunning   = "running"
	imageTaskStatusSucceeded = "succeeded"
	imageTaskStatusFailed    = "failed"
)

var errImageAsyncQueueFull = errors.New("image task queue is full")

type imageAsyncPreparedRequest struct {
	responsesReq   []byte
	responseFormat string
	prompt         string
	upscale        bool
	targetLongEdge int
	metadata       map[string]any
}

type imageAsyncRequestError struct {
	status  int
	message string
	typ     string
}

type imageAsyncTask struct {
	ID              string
	Status          string
	Progress        string
	CreatedAt       int64
	StartedAt       int64
	CompletedAt     int64
	UpdatedAt       int64
	ResponsesReq    []byte
	ResponseFormat  string
	Response        []byte
	ErrorMessage    string
	ErrorStatus     int
	Handler         *OpenAIAPIHandler
	Upscale         bool
	TargetLongEdge  int
	UpscaleMetadata map[string]any
	UpscaleJobIDs   []string
}

type imageAsyncTaskStore struct {
	mu      sync.RWMutex
	tasks   map[string]*imageAsyncTask
	queue   chan string
	timeout time.Duration
	ttl     time.Duration
}

var (
	imageAsyncStoreOnce sync.Once
	imageAsyncStore     *imageAsyncTaskStore
)

func (h *OpenAIAPIHandler) ImagesGenerationsAsync(c *gin.Context) {
	if imagesDisabled(h) {
		c.AbortWithStatus(http.StatusNotFound)
		return
	}
	if !imageAsyncTasksEnabled() {
		writeImageAsyncRequestError(c, &imageAsyncRequestError{
			status:  http.StatusServiceUnavailable,
			message: "image async tasks are not enabled",
			typ:     "service_unavailable",
		})
		return
	}

	rawJSON, err := c.GetRawData()
	if err != nil {
		writeImageAsyncRequestError(c, newImageAsyncInvalidRequest(fmt.Sprintf("Invalid request: %v", err)))
		return
	}
	prepared, reqErr := prepareImageGenerationAsyncRequest(rawJSON)
	if reqErr != nil {
		writeImageAsyncRequestError(c, reqErr)
		return
	}

	task, err := getImageAsyncTaskStore().enqueue(h, prepared)
	if err != nil {
		c.Header("Retry-After", "1")
		writeImageAsyncRequestError(c, &imageAsyncRequestError{
			status:  http.StatusServiceUnavailable,
			message: err.Error(),
			typ:     "server_error",
		})
		return
	}
	c.JSON(http.StatusAccepted, imageAsyncTaskResponse(task))
}

func (h *OpenAIAPIHandler) enqueueImageGenerationUpscaleTask(c *gin.Context, rawJSON []byte) {
	if !imageAsyncTasksEnabled() {
		writeImageAsyncRequestError(c, &imageAsyncRequestError{
			status:  http.StatusServiceUnavailable,
			message: "image async tasks are not enabled",
			typ:     "service_unavailable",
		})
		return
	}
	prepared, reqErr := prepareImageGenerationAsyncRequest(rawJSON)
	if reqErr != nil {
		writeImageAsyncRequestError(c, reqErr)
		return
	}
	task, err := getImageAsyncTaskStore().enqueue(h, prepared)
	if err != nil {
		c.Header("Retry-After", "1")
		writeImageAsyncRequestError(c, &imageAsyncRequestError{
			status:  http.StatusServiceUnavailable,
			message: err.Error(),
			typ:     "server_error",
		})
		return
	}
	c.JSON(http.StatusAccepted, imageAsyncTaskResponse(task))
}

func (h *OpenAIAPIHandler) ImagesEditsAsync(c *gin.Context) {
	if imagesDisabled(h) {
		c.AbortWithStatus(http.StatusNotFound)
		return
	}
	if !imageAsyncTasksEnabled() {
		writeImageAsyncRequestError(c, &imageAsyncRequestError{
			status:  http.StatusServiceUnavailable,
			message: "image async tasks are not enabled",
			typ:     "service_unavailable",
		})
		return
	}

	prepared, reqErr := prepareImageEditAsyncRequest(c)
	if reqErr != nil {
		writeImageAsyncRequestError(c, reqErr)
		return
	}
	task, err := getImageAsyncTaskStore().enqueue(h, prepared)
	if err != nil {
		c.Header("Retry-After", "1")
		writeImageAsyncRequestError(c, &imageAsyncRequestError{
			status:  http.StatusServiceUnavailable,
			message: err.Error(),
			typ:     "server_error",
		})
		return
	}
	c.JSON(http.StatusAccepted, imageAsyncTaskResponse(task))
}

func (h *OpenAIAPIHandler) GetImageTask(c *gin.Context) {
	taskID := strings.TrimSpace(c.Param("task_id"))
	task, ok := getImageAsyncTaskStore().snapshot(taskID)
	if !ok {
		writeImageAsyncRequestError(c, &imageAsyncRequestError{
			status:  http.StatusNotFound,
			message: "image task not found",
			typ:     "not_found_error",
		})
		return
	}
	c.JSON(http.StatusOK, imageAsyncTaskResponse(task))
}

func getImageAsyncTaskStore() *imageAsyncTaskStore {
	imageAsyncStoreOnce.Do(func() {
		workers := imageAsyncEnvInt("CPA_IMAGE_TASK_WORKERS", defaultImageAsyncWorkers, 1)
		queueSize := imageAsyncEnvInt("CPA_IMAGE_TASK_QUEUE_SIZE", defaultImageAsyncQueueSize, workers)
		timeoutSeconds := imageAsyncEnvInt("CPA_IMAGE_TASK_TIMEOUT_SECONDS", defaultImageAsyncTimeoutSeconds, 1)
		ttlSeconds := imageAsyncEnvInt("CPA_IMAGE_TASK_TTL_SECONDS", defaultImageAsyncTTLSeconds, 60)
		imageAsyncStore = &imageAsyncTaskStore{
			tasks:   make(map[string]*imageAsyncTask),
			queue:   make(chan string, queueSize),
			timeout: time.Duration(timeoutSeconds) * time.Second,
			ttl:     time.Duration(ttlSeconds) * time.Second,
		}
		for i := 0; i < workers; i++ {
			go imageAsyncStore.worker()
		}
		go imageAsyncStore.cleanupLoop()
		log.Infof("openai images async task store started, workers=%d queue=%d timeout=%s ttl=%s", workers, queueSize, imageAsyncStore.timeout, imageAsyncStore.ttl)
	})
	return imageAsyncStore
}

func (s *imageAsyncTaskStore) enqueue(h *OpenAIAPIHandler, prepared *imageAsyncPreparedRequest) (*imageAsyncTask, error) {
	if h == nil || h.BaseAPIHandler == nil || h.AuthManager == nil {
		return nil, errors.New("image task executor is not ready")
	}
	taskID := "task_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	now := time.Now().Unix()
	task := &imageAsyncTask{
		ID:              taskID,
		Status:          imageTaskStatusQueued,
		Progress:        "0%",
		CreatedAt:       now,
		UpdatedAt:       now,
		ResponsesReq:    append([]byte(nil), prepared.responsesReq...),
		ResponseFormat:  prepared.responseFormat,
		Handler:         h,
		Upscale:         prepared.upscale,
		TargetLongEdge:  prepared.targetLongEdge,
		UpscaleMetadata: cloneMap(prepared.metadata),
	}

	s.mu.Lock()
	s.tasks[task.ID] = task
	s.mu.Unlock()

	select {
	case s.queue <- task.ID:
		return s.snapshotOrTask(task.ID, task), nil
	default:
		s.mu.Lock()
		delete(s.tasks, task.ID)
		s.mu.Unlock()
		return nil, errImageAsyncQueueFull
	}
}

func (s *imageAsyncTaskStore) worker() {
	for taskID := range s.queue {
		s.run(taskID)
	}
}

func (s *imageAsyncTaskStore) run(taskID string) {
	task, ok := s.snapshot(taskID)
	if !ok {
		return
	}
	now := time.Now().Unix()
	s.update(taskID, func(t *imageAsyncTask) {
		t.Status = imageTaskStatusRunning
		t.Progress = "10%"
		t.StartedAt = now
		t.UpdatedAt = now
	})

	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	ctx = handlers.WithDisallowFreeAuth(ctx)

	out, errMsg := executeImageAsyncTask(ctx, task.Handler, task.ResponsesReq, task.ResponseFormat)
	if errMsg != nil {
		status := errMsg.StatusCode
		if status <= 0 {
			status = http.StatusInternalServerError
		}
		message := http.StatusText(status)
		if errMsg.Error != nil && strings.TrimSpace(errMsg.Error.Error()) != "" {
			message = errMsg.Error.Error()
		}
		finishedAt := time.Now().Unix()
		s.update(taskID, func(t *imageAsyncTask) {
			t.Status = imageTaskStatusFailed
			t.Progress = "100%"
			t.CompletedAt = finishedAt
			t.UpdatedAt = finishedAt
			t.ErrorStatus = status
			t.ErrorMessage = message
		})
		return
	}

	finishedAt := time.Now().Unix()
	var upscaleJobIDs []string
	if task.Upscale {
		generationSec := float64(finishedAt - task.StartedAt)
		jobs, err := createUpscaleJobsFromImageResponse(out, task.ID, task.ResponsesReq, task.TargetLongEdge, task.UpscaleMetadata, generationSec)
		if err != nil {
			s.update(taskID, func(t *imageAsyncTask) {
				t.Status = imageTaskStatusFailed
				t.Progress = "100%"
				t.CompletedAt = finishedAt
				t.UpdatedAt = finishedAt
				t.ErrorStatus = http.StatusInternalServerError
				t.ErrorMessage = err.Error()
			})
			return
		}
		upscaleJobIDs = jobs
	}
	s.update(taskID, func(t *imageAsyncTask) {
		t.Status = imageTaskStatusSucceeded
		t.Progress = "100%"
		t.CompletedAt = finishedAt
		t.UpdatedAt = finishedAt
		t.Response = append(t.Response[:0], out...)
		t.ErrorStatus = 0
		t.ErrorMessage = ""
		t.UpscaleJobIDs = append(t.UpscaleJobIDs[:0], upscaleJobIDs...)
	})
}

func executeImageAsyncTask(ctx context.Context, h *OpenAIAPIHandler, responsesReq []byte, responseFormat string) ([]byte, *interfaces.ErrorMessage) {
	mainModel := strings.TrimSpace(gjson.GetBytes(responsesReq, "model").String())
	if mainModel == "" {
		mainModel = defaultImagesMainModel
	}
	out, _, errMsg := h.collectImagesFromResponsesWithRetry(ctx, mainModel, responsesReq, responseFormat)
	return out, errMsg
}

func (s *imageAsyncTaskStore) snapshot(taskID string) (*imageAsyncTask, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	task, ok := s.tasks[taskID]
	if !ok || task == nil {
		return nil, false
	}
	clone := *task
	clone.ResponsesReq = append([]byte(nil), task.ResponsesReq...)
	clone.Response = append([]byte(nil), task.Response...)
	clone.UpscaleMetadata = cloneMap(task.UpscaleMetadata)
	clone.UpscaleJobIDs = append([]string(nil), task.UpscaleJobIDs...)
	return &clone, true
}

func (s *imageAsyncTaskStore) snapshotOrTask(taskID string, fallback *imageAsyncTask) *imageAsyncTask {
	if task, ok := s.snapshot(taskID); ok {
		return task
	}
	return fallback
}

func (s *imageAsyncTaskStore) update(taskID string, fn func(*imageAsyncTask)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	task, ok := s.tasks[taskID]
	if !ok || task == nil {
		return
	}
	fn(task)
}

func (s *imageAsyncTaskStore) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		cutoff := time.Now().Add(-s.ttl).Unix()
		s.mu.Lock()
		for id, task := range s.tasks {
			if task == nil {
				delete(s.tasks, id)
				continue
			}
			referenceTime := task.CompletedAt
			if referenceTime == 0 {
				referenceTime = task.UpdatedAt
			}
			if referenceTime == 0 {
				referenceTime = task.CreatedAt
			}
			if referenceTime > 0 && referenceTime < cutoff {
				delete(s.tasks, id)
			}
		}
		s.mu.Unlock()
	}
}

func imageAsyncTaskResponse(task *imageAsyncTask) gin.H {
	resp := gin.H{
		"id":           task.ID,
		"task_id":      task.ID,
		"object":       "image.task",
		"status":       task.Status,
		"progress":     task.Progress,
		"created_at":   task.CreatedAt,
		"started_at":   task.StartedAt,
		"completed_at": task.CompletedAt,
	}
	if task.Upscale {
		resp["upscale"] = true
	}
	var upscaleJobs []gin.H
	if len(task.UpscaleJobIDs) > 0 {
		upscaleJobs = make([]gin.H, 0, len(task.UpscaleJobIDs))
		for _, jobID := range task.UpscaleJobIDs {
			if job, ok := getUpscaleJobStore().snapshot(jobID); ok {
				upscaleJobs = append(upscaleJobs, publicUpscaleJob(job))
			} else {
				upscaleJobs = append(upscaleJobs, gin.H{"id": jobID, "status": "unknown"})
			}
		}
		resp["upscale_jobs"] = upscaleJobs
		if len(upscaleJobs) == 1 {
			resp["upscale_job"] = upscaleJobs[0]
			if url, _ := upscaleJobs[0]["result_image_url"].(string); strings.TrimSpace(url) != "" {
				resp["result_image_url"] = url
				resp["final_url"] = url
			}
		}
	}
	if task.Status == imageTaskStatusFailed {
		status := task.ErrorStatus
		if status <= 0 {
			status = http.StatusInternalServerError
		}
		message := task.ErrorMessage
		if strings.TrimSpace(message) == "" {
			message = http.StatusText(status)
		}
		resp["error"] = gin.H{
			"message": message,
			"type":    "server_error",
			"status":  status,
		}
		return resp
	}
	if task.Status != imageTaskStatusSucceeded || len(task.Response) == 0 {
		return resp
	}

	var payload map[string]any
	if err := json.Unmarshal(task.Response, &payload); err == nil {
		if task.Upscale {
			if finalData := finalImageDataFromUpscaleJobs(upscaleJobs); len(finalData) > 0 {
				resp["data"] = finalData
			}
			if created, ok := payload["created"]; ok {
				resp["created"] = created
			}
			if usage, ok := payload["usage"]; ok {
				resp["usage"] = usage
			}
			return resp
		}
		resp["response"] = payload
		if data, ok := payload["data"]; ok {
			resp["data"] = data
		}
		if created, ok := payload["created"]; ok {
			resp["created"] = created
		}
		if usage, ok := payload["usage"]; ok {
			resp["usage"] = usage
		}
	}
	return resp
}

func finalImageDataFromUpscaleJobs(jobs []gin.H) []gin.H {
	if len(jobs) == 0 {
		return nil
	}
	data := make([]gin.H, 0, len(jobs))
	for _, job := range jobs {
		if status, _ := job["status"].(string); status != "" && status != upscaleJobStatusSucceeded {
			return nil
		}
		url, _ := job["result_image_url"].(string)
		url = strings.TrimSpace(url)
		if url == "" {
			return nil
		}
		data = append(data, gin.H{"url": url})
	}
	return data
}

func prepareImageGenerationAsyncRequest(rawJSON []byte) (*imageAsyncPreparedRequest, *imageAsyncRequestError) {
	if !json.Valid(rawJSON) {
		return nil, newImageAsyncInvalidRequest("Invalid request: body must be valid JSON")
	}

	imageModel := strings.TrimSpace(gjson.GetBytes(rawJSON, "model").String())
	if imageModel == "" {
		imageModel = defaultImagesToolModel
	}
	if !isSupportedImagesModel(imageModel) {
		return nil, unsupportedImageAsyncModelError(imageModel)
	}

	prompt := strings.TrimSpace(gjson.GetBytes(rawJSON, "prompt").String())
	if prompt == "" {
		return nil, newImageAsyncInvalidRequest("Invalid request: prompt is required")
	}

	responseFormat := strings.TrimSpace(gjson.GetBytes(rawJSON, "response_format").String())
	upscale, targetLongEdge, metadata := parseImageAsyncUpscaleJSON(rawJSON)
	if upscale {
		if !imageObjectStorageConfigured() {
			return nil, &imageAsyncRequestError{
				status:  http.StatusServiceUnavailable,
				message: "image object storage is required for upscale jobs",
				typ:     "service_unavailable",
			}
		}
		responseFormat = "url"
	}

	tool := []byte(`{"type":"image_generation","action":"generate"}`)
	tool, _ = sjson.SetBytes(tool, "model", imageModel)

	for _, field := range []string{"size", "quality", "background", "output_format", "moderation"} {
		if v := strings.TrimSpace(gjson.GetBytes(rawJSON, field).String()); v != "" {
			tool, _ = sjson.SetBytes(tool, field, v)
		}
	}
	for _, field := range []string{"output_compression", "partial_images"} {
		if v := gjson.GetBytes(rawJSON, field); v.Exists() && v.Type == gjson.Number {
			tool, _ = sjson.SetBytes(tool, field, v.Int())
		}
	}

	return &imageAsyncPreparedRequest{
		responsesReq:   buildImagesResponsesRequest(prompt, nil, tool),
		responseFormat: responseFormat,
		prompt:         prompt,
		upscale:        upscale,
		targetLongEdge: targetLongEdge,
		metadata:       metadata,
	}, nil
}

func prepareImageEditAsyncRequest(c *gin.Context) (*imageAsyncPreparedRequest, *imageAsyncRequestError) {
	contentType := strings.ToLower(strings.TrimSpace(c.GetHeader("Content-Type")))
	if strings.HasPrefix(contentType, "application/json") {
		rawJSON, err := c.GetRawData()
		if err != nil {
			return nil, newImageAsyncInvalidRequest(fmt.Sprintf("Invalid request: %v", err))
		}
		return prepareImageEditAsyncJSONRequest(rawJSON)
	}
	if strings.HasPrefix(contentType, "multipart/form-data") || contentType == "" {
		return prepareImageEditAsyncMultipartRequest(c)
	}
	return nil, newImageAsyncInvalidRequest(fmt.Sprintf("Invalid request: unsupported Content-Type %q", contentType))
}

func prepareImageEditAsyncJSONRequest(rawJSON []byte) (*imageAsyncPreparedRequest, *imageAsyncRequestError) {
	if !json.Valid(rawJSON) {
		return nil, newImageAsyncInvalidRequest("Invalid request: body must be valid JSON")
	}

	imageModel := strings.TrimSpace(gjson.GetBytes(rawJSON, "model").String())
	if imageModel == "" {
		imageModel = defaultImagesToolModel
	}
	if !isSupportedImagesModel(imageModel) {
		return nil, unsupportedImageAsyncModelError(imageModel)
	}

	prompt := strings.TrimSpace(gjson.GetBytes(rawJSON, "prompt").String())
	if prompt == "" {
		return nil, newImageAsyncInvalidRequest("Invalid request: prompt is required")
	}

	var images []string
	imagesResult := gjson.GetBytes(rawJSON, "images")
	if imagesResult.IsArray() {
		for _, img := range imagesResult.Array() {
			url := strings.TrimSpace(img.Get("image_url").String())
			if url == "" {
				continue
			}
			images = append(images, url)
		}
	}
	if len(images) == 0 {
		return nil, newImageAsyncInvalidRequest("Invalid request: images[].image_url is required (file_id is not supported)")
	}

	var maskDataURL *string
	if mask := gjson.GetBytes(rawJSON, "mask.image_url"); mask.Exists() {
		url := strings.TrimSpace(mask.String())
		if url != "" {
			maskDataURL = &url
		}
	} else if mask := gjson.GetBytes(rawJSON, "mask.file_id"); mask.Exists() {
		return nil, newImageAsyncInvalidRequest("Invalid request: mask.file_id is not supported (use mask.image_url instead)")
	}

	responseFormat := strings.TrimSpace(gjson.GetBytes(rawJSON, "response_format").String())
	upscale, targetLongEdge, metadata := parseImageAsyncUpscaleJSON(rawJSON)
	if upscale {
		if !imageObjectStorageConfigured() {
			return nil, &imageAsyncRequestError{
				status:  http.StatusServiceUnavailable,
				message: "image object storage is required for upscale jobs",
				typ:     "service_unavailable",
			}
		}
		responseFormat = "url"
	}

	tool := []byte(`{"type":"image_generation","action":"edit"}`)
	tool, _ = sjson.SetBytes(tool, "model", imageModel)

	for _, field := range []string{"size", "quality", "background", "output_format", "input_fidelity", "moderation"} {
		if v := strings.TrimSpace(gjson.GetBytes(rawJSON, field).String()); v != "" {
			tool, _ = sjson.SetBytes(tool, field, v)
		}
	}
	for _, field := range []string{"output_compression", "partial_images"} {
		if v := gjson.GetBytes(rawJSON, field); v.Exists() && v.Type == gjson.Number {
			tool, _ = sjson.SetBytes(tool, field, v.Int())
		}
	}
	if maskDataURL != nil && strings.TrimSpace(*maskDataURL) != "" {
		tool, _ = sjson.SetBytes(tool, "input_image_mask.image_url", strings.TrimSpace(*maskDataURL))
	}

	return &imageAsyncPreparedRequest{
		responsesReq:   buildImagesResponsesRequest(prompt, images, tool),
		responseFormat: responseFormat,
		prompt:         prompt,
		upscale:        upscale,
		targetLongEdge: targetLongEdge,
		metadata:       metadata,
	}, nil
}

func prepareImageEditAsyncMultipartRequest(c *gin.Context) (*imageAsyncPreparedRequest, *imageAsyncRequestError) {
	form, err := c.MultipartForm()
	if err != nil {
		return nil, newImageAsyncInvalidRequest(fmt.Sprintf("Invalid request: %v", err))
	}
	if form == nil {
		return nil, newImageAsyncInvalidRequest("Invalid request: image is required")
	}

	imageModel := strings.TrimSpace(c.PostForm("model"))
	if imageModel == "" {
		imageModel = defaultImagesToolModel
	}
	if !isSupportedImagesModel(imageModel) {
		return nil, unsupportedImageAsyncModelError(imageModel)
	}

	prompt := strings.TrimSpace(c.PostForm("prompt"))
	if prompt == "" {
		return nil, newImageAsyncInvalidRequest("Invalid request: prompt is required")
	}

	var imageFiles []string
	if files := form.File["image[]"]; len(files) > 0 {
		for _, fh := range files {
			dataURL, err := multipartFileToDataURL(fh)
			if err != nil {
				return nil, newImageAsyncInvalidRequest(fmt.Sprintf("Invalid request: %v", err))
			}
			imageFiles = append(imageFiles, dataURL)
		}
	} else if files := form.File["image"]; len(files) > 0 {
		for _, fh := range files {
			dataURL, err := multipartFileToDataURL(fh)
			if err != nil {
				return nil, newImageAsyncInvalidRequest(fmt.Sprintf("Invalid request: %v", err))
			}
			imageFiles = append(imageFiles, dataURL)
		}
	}
	if len(imageFiles) == 0 {
		return nil, newImageAsyncInvalidRequest("Invalid request: image is required")
	}

	var maskDataURL *string
	if maskFiles := form.File["mask"]; len(maskFiles) > 0 && maskFiles[0] != nil {
		dataURL, err := multipartFileToDataURL(maskFiles[0])
		if err != nil {
			return nil, newImageAsyncInvalidRequest(fmt.Sprintf("Invalid request: %v", err))
		}
		maskDataURL = &dataURL
	}

	responseFormat := strings.TrimSpace(c.PostForm("response_format"))
	upscale, targetLongEdge := parseImageAsyncUpscaleForm(c)
	if upscale {
		if !imageObjectStorageConfigured() {
			return nil, &imageAsyncRequestError{
				status:  http.StatusServiceUnavailable,
				message: "image object storage is required for upscale jobs",
				typ:     "service_unavailable",
			}
		}
		responseFormat = "url"
	}

	tool := []byte(`{"type":"image_generation","action":"edit"}`)
	tool, _ = sjson.SetBytes(tool, "model", imageModel)

	for _, field := range []string{"size", "quality", "background", "output_format", "input_fidelity", "moderation"} {
		if v := strings.TrimSpace(c.PostForm(field)); v != "" {
			tool, _ = sjson.SetBytes(tool, field, v)
		}
	}
	for _, field := range []string{"output_compression", "partial_images"} {
		if v := strings.TrimSpace(c.PostForm(field)); v != "" {
			tool, _ = sjson.SetBytes(tool, field, parseIntField(v, 0))
		}
	}
	if maskDataURL != nil && strings.TrimSpace(*maskDataURL) != "" {
		tool, _ = sjson.SetBytes(tool, "input_image_mask.image_url", strings.TrimSpace(*maskDataURL))
	}

	return &imageAsyncPreparedRequest{
		responsesReq:   buildImagesResponsesRequest(prompt, imageFiles, tool),
		responseFormat: responseFormat,
		prompt:         prompt,
		upscale:        upscale,
		targetLongEdge: targetLongEdge,
		metadata:       nil,
	}, nil
}

func createUpscaleJobsFromImageResponse(response []byte, imageTaskID string, responsesReq []byte, targetLongEdge int, metadata map[string]any, generationSec float64) ([]string, error) {
	if targetLongEdge <= 0 {
		targetLongEdge = defaultUpscaleTargetLongEdge
	}
	root := gjson.ParseBytes(response)
	data := root.Get("data")
	if !data.IsArray() || len(data.Array()) == 0 {
		return nil, errors.New("image response has no data items for upscale")
	}
	sourceWidth, sourceHeight := parseImageSize(root.Get("size").String())
	if sourceWidth == nil || sourceHeight == nil {
		sourceWidth, sourceHeight = parseImageSize(gjson.GetBytes(responsesReq, "tools.0.size").String())
	}
	prompt := strings.TrimSpace(gjson.GetBytes(responsesReq, "input.0.content.0.text").String())
	jobIDs := make([]string, 0, len(data.Array()))
	for index, item := range data.Array() {
		sourceURL := strings.TrimSpace(item.Get("url").String())
		if sourceURL == "" {
			return nil, errors.New("image response item has no source URL for upscale")
		}
		if strings.HasPrefix(sourceURL, "data:") {
			return nil, errors.New("upscale source URL is a data URL; configure image R2 object storage")
		}
		jobMetadata := cloneMap(metadata)
		if jobMetadata == nil {
			jobMetadata = make(map[string]any)
		}
		jobMetadata["image_task_id"] = imageTaskID
		jobMetadata["image_index"] = index
		if objectKey := strings.TrimSpace(item.Get("object_key").String()); objectKey != "" {
			jobMetadata["source_object_key"] = objectKey
		}
		job, err := getUpscaleJobStore().create(upscaleJobCreateRequest{
			SourceImageURL: sourceURL,
			SourceWidth:    sourceWidth,
			SourceHeight:   sourceHeight,
			TargetLongEdge: targetLongEdge,
			Prompt:         prompt,
			Metadata:       jobMetadata,
			GenerationSec:  generationSec,
		})
		if err != nil {
			return nil, err
		}
		jobIDs = append(jobIDs, job.ID)
	}
	return jobIDs, nil
}

func parseImageSize(size string) (any, any) {
	parts := strings.Split(strings.ToLower(strings.TrimSpace(size)), "x")
	if len(parts) != 2 {
		return nil, nil
	}
	w, errW := strconv.Atoi(strings.TrimSpace(parts[0]))
	h, errH := strconv.Atoi(strings.TrimSpace(parts[1]))
	if errW != nil || errH != nil || w <= 0 || h <= 0 {
		return nil, nil
	}
	return w, h
}

func parseImageAsyncUpscaleJSON(rawJSON []byte) (bool, int, map[string]any) {
	root := gjson.ParseBytes(rawJSON)
	enabled := false
	if v := root.Get("upscale"); v.Exists() {
		switch {
		case v.IsObject():
			enabled = parseBoolField(v.Get("enabled").String(), true)
		case v.Type == gjson.True:
			enabled = true
		case v.Type == gjson.String:
			enabled = parseBoolField(v.String(), false)
		}
	}
	for _, field := range []string{"upscale_4k", "enable_upscale"} {
		if v := root.Get(field); v.Exists() {
			enabled = v.Bool()
		}
	}
	target := defaultUpscaleTargetLongEdge
	for _, field := range []string{"upscale.target_long_edge", "upscale.long_edge", "target_long_edge", "upscale_target_long_edge"} {
		if v := root.Get(field); v.Exists() && v.Type == gjson.Number && v.Int() > 0 {
			target = int(v.Int())
			break
		}
	}
	var metadata map[string]any
	if raw := root.Get("upscale.metadata"); raw.Exists() && raw.IsObject() {
		_ = json.Unmarshal([]byte(raw.Raw), &metadata)
	}
	return enabled, target, metadata
}

func parseImageAsyncUpscaleForm(c *gin.Context) (bool, int) {
	enabled := parseBoolField(c.PostForm("upscale"), false)
	if c.PostForm("upscale_4k") != "" {
		enabled = parseBoolField(c.PostForm("upscale_4k"), enabled)
	}
	target := defaultUpscaleTargetLongEdge
	for _, field := range []string{"target_long_edge", "upscale_target_long_edge"} {
		if raw := strings.TrimSpace(c.PostForm(field)); raw != "" {
			if v, err := strconv.Atoi(raw); err == nil && v > 0 {
				target = v
				break
			}
		}
	}
	return enabled, target
}

func imagesDisabled(h *OpenAIAPIHandler) bool {
	return h != nil &&
		h.BaseAPIHandler != nil &&
		h.BaseAPIHandler.Cfg != nil &&
		h.BaseAPIHandler.Cfg.DisableImageGeneration == internalconfig.DisableImageGenerationAll
}

func imageAsyncTasksEnabled() bool {
	raw, ok := os.LookupEnv("CPA_IMAGE_TASKS_ENABLED")
	if !ok || strings.TrimSpace(raw) == "" {
		return true
	}
	return parseBoolField(raw, true)
}

func imageAsyncEnvInt(name string, fallback int, min int) int {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	if v < min {
		return min
	}
	return v
}

func unsupportedImageAsyncModelError(model string) *imageAsyncRequestError {
	return newImageAsyncInvalidRequest(fmt.Sprintf("Model %s is not supported on %s or %s. Use %s.", model, imagesGenerationsPath, imagesEditsPath, defaultImagesToolModel))
}

func newImageAsyncInvalidRequest(message string) *imageAsyncRequestError {
	return &imageAsyncRequestError{
		status:  http.StatusBadRequest,
		message: message,
		typ:     "invalid_request_error",
	}
}

func writeImageAsyncRequestError(c *gin.Context, err *imageAsyncRequestError) {
	if err == nil {
		err = &imageAsyncRequestError{
			status:  http.StatusInternalServerError,
			message: http.StatusText(http.StatusInternalServerError),
			typ:     "server_error",
		}
	}
	status := err.status
	if status <= 0 {
		status = http.StatusInternalServerError
	}
	typ := strings.TrimSpace(err.typ)
	if typ == "" {
		typ = "invalid_request_error"
	}
	c.JSON(status, handlers.ErrorResponse{
		Error: handlers.ErrorDetail{
			Message: err.message,
			Type:    typ,
		},
	})
}
