package openai

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/tidwall/gjson"
)

func resetUpscaleJobStoreForTest(t *testing.T) {
	t.Helper()
	t.Setenv("CPA_UPSCALE_JOBS_FILE", t.TempDir()+"/jobs.json")
	t.Setenv("CPA_UPSCALE_WORKER_SECRET", "worker-secret")
	upscaleStoreOnce = sync.Once{}
	upscaleStore = nil
	t.Cleanup(func() {
		upscaleStoreOnce = sync.Once{}
		upscaleStore = nil
	})
}

func TestUpscaleJobStoreClaimAndComplete(t *testing.T) {
	resetUpscaleJobStoreForTest(t)

	store := getUpscaleJobStore()
	created, err := store.create(upscaleJobCreateRequest{
		SourceImageURL: "https://cdn.example.com/source.png",
		SourceWidth:    2048,
		SourceHeight:   1360,
		TargetLongEdge: 4096,
		Prompt:         "poster",
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}

	claimed, ok, err := store.claim("mac-1")
	if err != nil {
		t.Fatalf("claim job: %v", err)
	}
	if !ok || claimed.ID != created.ID {
		t.Fatalf("claimed job = %#v, ok=%t; want %s", claimed, ok, created.ID)
	}
	if claimed.Status != upscaleJobStatusRunning || claimed.Attempts != 1 || claimed.WorkerID != "mac-1" {
		t.Fatalf("unexpected claimed job: %#v", claimed)
	}

	completed, err := store.workerUpdate(created.ID, "mac-1", "complete", map[string]any{
		"result_image_url": "https://cdn.example.com/final.png",
		"output_width":     4096,
		"output_height":    2720,
		"upscale_sec":      66.5,
	})
	if err != nil {
		t.Fatalf("complete job: %v", err)
	}
	if completed.Status != upscaleJobStatusSucceeded {
		t.Fatalf("status = %q, want %q", completed.Status, upscaleJobStatusSucceeded)
	}
	if completed.ResultImageURL != "https://cdn.example.com/final.png" {
		t.Fatalf("result url = %q", completed.ResultImageURL)
	}

	upscaleStoreOnce = sync.Once{}
	upscaleStore = nil
	reloaded, ok := getUpscaleJobStore().snapshot(created.ID)
	if !ok {
		t.Fatal("reloaded job not found")
	}
	if reloaded.ResultImageURL != completed.ResultImageURL {
		t.Fatalf("reloaded result url = %q", reloaded.ResultImageURL)
	}
}

func TestCreateUpscaleJobsFromImageResponse(t *testing.T) {
	resetUpscaleJobStoreForTest(t)

	response := []byte(`{"created":123,"size":"2048x1360","data":[{"url":"https://cdn.example.com/source.png","object_key":"images/source.png"}]}`)
	tool := []byte(`{"type":"image_generation","action":"generate","model":"gpt-image-2","size":"2048x1360"}`)
	responsesReq := buildImagesResponsesRequest("western poster", nil, tool)

	ids, err := createUpscaleJobsFromImageResponse(response, "task_123", responsesReq, 4096, map[string]any{"user": "u1"}, 12.3)
	if err != nil {
		t.Fatalf("create upscale jobs: %v", err)
	}
	if len(ids) != 1 {
		t.Fatalf("job count = %d, want 1", len(ids))
	}
	job, ok := getUpscaleJobStore().snapshot(ids[0])
	if !ok {
		t.Fatal("job not found")
	}
	if job.SourceImageURL != "https://cdn.example.com/source.png" {
		t.Fatalf("source url = %q", job.SourceImageURL)
	}
	if job.SourceWidth != 2048 || job.SourceHeight != 1360 {
		t.Fatalf("source dims = %v x %v", job.SourceWidth, job.SourceHeight)
	}
	raw, _ := jsonMarshalForTest(job.Metadata)
	if got := gjson.GetBytes(raw, "image_task_id").String(); got != "task_123" {
		t.Fatalf("metadata image_task_id = %q", got)
	}
	if got := gjson.GetBytes(raw, "source_object_key").String(); got != "images/source.png" {
		t.Fatalf("metadata source_object_key = %q", got)
	}
}

func jsonMarshalForTest(v any) ([]byte, error) {
	return json.Marshal(v)
}
