package openai

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	internalconfig "github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
	"github.com/tidwall/gjson"
)

func resetImageObjectEnvForTest(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		"CPA_IMAGE_R2_ENDPOINT",
		"CPA_IMAGE_OBJECT_STORAGE_ENABLED",
		"CPA_IMAGE_R2_ENABLED",
		"CPA_R2_ENDPOINT",
		"R2_ENDPOINT",
		"CPA_IMAGE_R2_BUCKET",
		"CPA_R2_BUCKET",
		"R2_BUCKET",
		"CPA_IMAGE_R2_ACCESS_KEY_ID",
		"CPA_IMAGE_R2_ACCESS_KEY",
		"CPA_R2_ACCESS_KEY_ID",
		"R2_ACCESS_KEY_ID",
		"AWS_ACCESS_KEY_ID",
		"CPA_IMAGE_R2_SECRET_ACCESS_KEY",
		"CPA_IMAGE_R2_SECRET_KEY",
		"CPA_R2_SECRET_ACCESS_KEY",
		"R2_SECRET_ACCESS_KEY",
		"AWS_SECRET_ACCESS_KEY",
		"CPA_IMAGE_DEFAULT_RESPONSE_FORMAT",
	} {
		t.Setenv(name, "")
	}
	resetImageObjectStorageForTest()
	t.Cleanup(resetImageObjectStorageForTest)
}

func performImagesEndpointRequest(t *testing.T, endpointPath string, contentType string, body io.Reader, handler gin.HandlerFunc) *httptest.ResponseRecorder {
	t.Helper()

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST(endpointPath, handler)

	req := httptest.NewRequest(http.MethodPost, endpointPath, body)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	return resp
}

func assertUnsupportedImagesModelResponse(t *testing.T, resp *httptest.ResponseRecorder, model string) {
	t.Helper()

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusBadRequest, resp.Body.String())
	}

	message := gjson.GetBytes(resp.Body.Bytes(), "error.message").String()
	expectedMessage := "Model " + model + " is not supported on " + imagesGenerationsPath + " or " + imagesEditsPath + ". Use " + defaultImagesToolModel + "."
	if message != expectedMessage {
		t.Fatalf("error message = %q, want %q", message, expectedMessage)
	}
	if errorType := gjson.GetBytes(resp.Body.Bytes(), "error.type").String(); errorType != "invalid_request_error" {
		t.Fatalf("error type = %q, want invalid_request_error", errorType)
	}
}

type retryImageStreamExecutor struct {
	mu    sync.Mutex
	calls int
}

func (e *retryImageStreamExecutor) Identifier() string { return "codex" }

func (e *retryImageStreamExecutor) Execute(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, errors.New("not implemented")
}

func (e *retryImageStreamExecutor) ExecuteStream(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (*coreexecutor.StreamResult, error) {
	e.mu.Lock()
	e.calls++
	call := e.calls
	e.mu.Unlock()

	ch := make(chan coreexecutor.StreamChunk, 2)
	if call == 1 {
		ch <- coreexecutor.StreamChunk{Payload: []byte(`data: {"type":"response.image_generation_call.partial_image","partial_image_b64":"AA==","output_format":"png"}` + "\n\n")}
		close(ch)
		return &coreexecutor.StreamResult{Chunks: ch}, nil
	}
	ch <- coreexecutor.StreamChunk{Payload: []byte(`data: {"type":"response.completed","response":{"created_at":123,"output":[{"type":"image_generation_call","result":"AA==","output_format":"png","size":"1024x1024","quality":"medium"}]}}` + "\n\n")}
	close(ch)
	return &coreexecutor.StreamResult{Chunks: ch}, nil
}

func (e *retryImageStreamExecutor) Refresh(ctx context.Context, auth *coreauth.Auth) (*coreauth.Auth, error) {
	return auth, nil
}

func (e *retryImageStreamExecutor) CountTokens(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, errors.New("not implemented")
}

func (e *retryImageStreamExecutor) HttpRequest(context.Context, *coreauth.Auth, *http.Request) (*http.Response, error) {
	return nil, errors.New("not implemented")
}

func (e *retryImageStreamExecutor) Calls() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.calls
}

func TestImagesModelValidationAllowsGPTImage2WithOptionalPrefix(t *testing.T) {
	for _, model := range []string{"gpt-image-2", "codex/gpt-image-2"} {
		if !isSupportedImagesModel(model) {
			t.Fatalf("expected %s to be supported", model)
		}
	}
	if isSupportedImagesModel("gpt-5.4-mini") {
		t.Fatal("expected gpt-5.4-mini to be rejected")
	}
}

func TestNormalizeImagesResponseFormatDefaultsToB64WithoutStorage(t *testing.T) {
	resetImageObjectEnvForTest(t)

	if got := normalizeImagesResponseFormat(""); got != "b64_json" {
		t.Fatalf("response format = %q, want b64_json", got)
	}
}

func TestNormalizeImagesResponseFormatDefaultsToURLWithStorage(t *testing.T) {
	resetImageObjectEnvForTest(t)
	t.Setenv("CPA_IMAGE_R2_ENDPOINT", "https://example.r2.cloudflarestorage.com")
	t.Setenv("CPA_IMAGE_R2_BUCKET", "images")
	t.Setenv("CPA_IMAGE_R2_ACCESS_KEY_ID", "access")
	t.Setenv("CPA_IMAGE_R2_SECRET_ACCESS_KEY", "secret")
	resetImageObjectStorageForTest()

	if got := normalizeImagesResponseFormat(""); got != "url" {
		t.Fatalf("response format = %q, want url", got)
	}
}

func TestNormalizeImagesResponseFormatCanDisableObjectStorage(t *testing.T) {
	resetImageObjectEnvForTest(t)
	t.Setenv("CPA_IMAGE_OBJECT_STORAGE_ENABLED", "false")
	t.Setenv("CPA_IMAGE_R2_ENDPOINT", "https://example.r2.cloudflarestorage.com")
	t.Setenv("CPA_IMAGE_R2_BUCKET", "images")
	t.Setenv("CPA_IMAGE_R2_ACCESS_KEY_ID", "access")
	t.Setenv("CPA_IMAGE_R2_SECRET_ACCESS_KEY", "secret")
	resetImageObjectStorageForTest()

	if got := normalizeImagesResponseFormat(""); got != "b64_json" {
		t.Fatalf("response format = %q, want b64_json", got)
	}
}

func TestBuildImagesAPIResponseURLFallsBackToDataURLWithoutStorage(t *testing.T) {
	resetImageObjectEnvForTest(t)

	out, err := buildImagesAPIResponse(context.Background(), []imageCallResult{
		{Result: "AA==", OutputFormat: "png", RevisedPrompt: "tiny"},
	}, 123, nil, imageCallResult{}, "url")
	if err != nil {
		t.Fatalf("buildImagesAPIResponse returned error: %v", err)
	}
	if got := gjson.GetBytes(out, "data.0.url").String(); got != "data:image/png;base64,AA==" {
		t.Fatalf("url = %q, want data URL fallback", got)
	}
	if got := gjson.GetBytes(out, "data.0.revised_prompt").String(); got != "tiny" {
		t.Fatalf("revised_prompt = %q, want tiny", got)
	}
}

func TestCollectImagesFromResponsesStreamRecoversOutputItemDoneOnDisconnect(t *testing.T) {
	resetImageObjectEnvForTest(t)
	data := make(chan []byte, 1)
	errs := make(chan *interfaces.ErrorMessage)
	data <- []byte(`data: {"type":"response.output_item.done","item":{"type":"image_generation_call","result":"AA==","output_format":"png","size":"1024x1024","quality":"medium"}}` + "\n\n")
	close(data)
	close(errs)

	out, errMsg := collectImagesFromResponsesStream(context.Background(), data, errs, "b64_json")
	if errMsg != nil {
		t.Fatalf("collectImagesFromResponsesStream returned error: %+v", errMsg)
	}
	if got := gjson.GetBytes(out, "data.0.b64_json").String(); got != "AA==" {
		t.Fatalf("b64_json = %q, want AA==", got)
	}
	if got := gjson.GetBytes(out, "size").String(); got != "1024x1024" {
		t.Fatalf("size = %q, want 1024x1024", got)
	}
}

func TestCollectImagesFromResponsesWithRetryRetriesAfterStreamDisconnect(t *testing.T) {
	resetImageObjectEnvForTest(t)
	t.Setenv("CPA_IMAGE_STREAM_MAX_ATTEMPTS", "2")
	t.Setenv("CPA_IMAGE_STREAM_RETRY_DELAY_MS", "0")

	executor := &retryImageStreamExecutor{}
	manager := coreauth.NewManager(nil, nil, nil)
	manager.RegisterExecutor(executor)
	auth := &coreauth.Auth{
		ID:       "auth-retry",
		Provider: "codex",
		Status:   coreauth.StatusActive,
		Metadata: map[string]any{"email": "retry@example.com"},
	}
	if _, err := manager.Register(context.Background(), auth); err != nil {
		t.Fatalf("manager.Register: %v", err)
	}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{ID: defaultImagesMainModel}})
	t.Cleanup(func() {
		registry.GetGlobalRegistry().UnregisterClient(auth.ID)
	})

	handler := NewOpenAIAPIHandler(handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))
	responsesReq := []byte(`{"model":"` + defaultImagesMainModel + `","stream":true}`)
	out, _, errMsg := handler.collectImagesFromResponsesWithRetry(context.Background(), defaultImagesMainModel, responsesReq, "b64_json")
	if errMsg != nil {
		t.Fatalf("collectImagesFromResponsesWithRetry returned error: %+v", errMsg)
	}
	if got := executor.Calls(); got != 2 {
		t.Fatalf("executor calls = %d, want 2", got)
	}
	if got := gjson.GetBytes(out, "data.0.b64_json").String(); got != "AA==" {
		t.Fatalf("b64_json = %q, want AA==", got)
	}
}

func TestNormalizeImageObjectPublicBaseAddsHTTPS(t *testing.T) {
	if got := normalizeImageObjectPublicBase("image.minotoken.xyz/"); got != "https://image.minotoken.xyz" {
		t.Fatalf("public base = %q, want https://image.minotoken.xyz", got)
	}
	if got := normalizeImageObjectPublicBase("https://image.minotoken.xyz/"); got != "https://image.minotoken.xyz" {
		t.Fatalf("public base = %q, want https://image.minotoken.xyz", got)
	}
}

func TestImagesGenerationsRejectsUnsupportedModel(t *testing.T) {
	handler := &OpenAIAPIHandler{}
	body := strings.NewReader(`{"model":"gpt-5.4-mini","prompt":"draw a square"}`)

	resp := performImagesEndpointRequest(t, imagesGenerationsPath, "application/json", body, handler.ImagesGenerations)

	assertUnsupportedImagesModelResponse(t, resp, "gpt-5.4-mini")
}

func TestImagesEditsJSONRejectsUnsupportedModel(t *testing.T) {
	handler := &OpenAIAPIHandler{}
	body := strings.NewReader(`{"model":"gpt-5.4-mini","prompt":"edit this","images":[{"image_url":"data:image/png;base64,AA=="}]}`)

	resp := performImagesEndpointRequest(t, imagesEditsPath, "application/json", body, handler.ImagesEdits)

	assertUnsupportedImagesModelResponse(t, resp, "gpt-5.4-mini")
}

func TestImagesEditsMultipartRejectsUnsupportedModel(t *testing.T) {
	handler := &OpenAIAPIHandler{}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if err := writer.WriteField("model", "gpt-5.4-mini"); err != nil {
		t.Fatalf("write model field: %v", err)
	}
	if err := writer.WriteField("prompt", "edit this"); err != nil {
		t.Fatalf("write prompt field: %v", err)
	}
	if errClose := writer.Close(); errClose != nil {
		t.Fatalf("close multipart writer: %v", errClose)
	}

	resp := performImagesEndpointRequest(t, imagesEditsPath, writer.FormDataContentType(), &body, handler.ImagesEdits)

	assertUnsupportedImagesModelResponse(t, resp, "gpt-5.4-mini")
}

func TestImagesGenerations_DisableImageGeneration_Returns404(t *testing.T) {
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{DisableImageGeneration: internalconfig.DisableImageGenerationAll}, nil)
	handler := NewOpenAIAPIHandler(base)
	body := strings.NewReader(`{"prompt":"draw a square"}`)

	resp := performImagesEndpointRequest(t, imagesGenerationsPath, "application/json", body, handler.ImagesGenerations)

	if resp.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusNotFound, resp.Body.String())
	}
}

func TestImagesEdits_DisableImageGeneration_Returns404(t *testing.T) {
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{DisableImageGeneration: internalconfig.DisableImageGenerationAll}, nil)
	handler := NewOpenAIAPIHandler(base)
	body := strings.NewReader(`{"prompt":"edit this","images":[{"image_url":"data:image/png;base64,AA=="}]}`)

	resp := performImagesEndpointRequest(t, imagesEditsPath, "application/json", body, handler.ImagesEdits)

	if resp.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusNotFound, resp.Body.String())
	}
}

func TestImagesGenerations_DisableImageGenerationChat_DoesNotReturn404(t *testing.T) {
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{DisableImageGeneration: internalconfig.DisableImageGenerationChat}, nil)
	handler := NewOpenAIAPIHandler(base)
	body := strings.NewReader(`{"model":"gpt-5.4-mini","prompt":"draw a square"}`)

	resp := performImagesEndpointRequest(t, imagesGenerationsPath, "application/json", body, handler.ImagesGenerations)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusBadRequest, resp.Body.String())
	}
}

func TestImagesEdits_DisableImageGenerationChat_DoesNotReturn404(t *testing.T) {
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{DisableImageGeneration: internalconfig.DisableImageGenerationChat}, nil)
	handler := NewOpenAIAPIHandler(base)
	body := strings.NewReader(`{"model":"gpt-5.4-mini","prompt":"edit this","images":[{"image_url":"data:image/png;base64,AA=="}]}`)

	resp := performImagesEndpointRequest(t, imagesEditsPath, "application/json", body, handler.ImagesEdits)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusBadRequest, resp.Body.String())
	}
}
