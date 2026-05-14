package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/tidwall/sjson"
)

const defaultImageObjectURLTTL = 7 * 24 * time.Hour

var errImageObjectStorageNotConfigured = errors.New("image object storage is not configured")

type imageObjectStorageConfig struct {
	Endpoint   string
	Bucket     string
	AccessKey  string
	SecretKey  string
	Region     string
	Prefix     string
	PublicBase string
	UseSSL     bool
}

type imageObjectStorage struct {
	client *minio.Client
	cfg    imageObjectStorageConfig
}

type storedImageObject struct {
	Key string
	URL string
}

var (
	imageObjectStorageOnce sync.Once
	imageObjectStorageInst *imageObjectStorage
	imageObjectStorageErr  error
)

func getImageObjectStorage() (*imageObjectStorage, error) {
	imageObjectStorageOnce.Do(func() {
		cfg, err := loadImageObjectStorageConfig()
		if err != nil {
			imageObjectStorageErr = err
			return
		}
		client, err := minio.New(cfg.Endpoint, &minio.Options{
			Creds:        credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
			Secure:       cfg.UseSSL,
			Region:       cfg.Region,
			BucketLookup: minio.BucketLookupPath,
		})
		if err != nil {
			imageObjectStorageErr = fmt.Errorf("image object storage: create client: %w", err)
			return
		}
		imageObjectStorageInst = &imageObjectStorage{client: client, cfg: cfg}
	})
	return imageObjectStorageInst, imageObjectStorageErr
}

func imageObjectStorageConfigured() bool {
	_, err := getImageObjectStorage()
	return err == nil
}

func loadImageObjectStorageConfig() (imageObjectStorageConfig, error) {
	rawEndpoint := firstImageObjectEnv("CPA_IMAGE_R2_ENDPOINT", "CPA_R2_ENDPOINT", "R2_ENDPOINT")
	bucket := firstImageObjectEnv("CPA_IMAGE_R2_BUCKET", "CPA_R2_BUCKET", "R2_BUCKET")
	accessKey := firstImageObjectEnv("CPA_IMAGE_R2_ACCESS_KEY_ID", "CPA_IMAGE_R2_ACCESS_KEY", "CPA_R2_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID")
	secretKey := firstImageObjectEnv("CPA_IMAGE_R2_SECRET_ACCESS_KEY", "CPA_IMAGE_R2_SECRET_KEY", "CPA_R2_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY")
	if rawEndpoint == "" || bucket == "" || accessKey == "" || secretKey == "" {
		return imageObjectStorageConfig{}, errImageObjectStorageNotConfigured
	}

	endpoint, useSSL, err := normalizeImageObjectEndpoint(rawEndpoint)
	if err != nil {
		return imageObjectStorageConfig{}, err
	}
	if rawUseSSL := strings.TrimSpace(firstImageObjectEnv("CPA_IMAGE_R2_USE_SSL", "CPA_R2_USE_SSL", "R2_USE_SSL")); rawUseSSL != "" {
		useSSL = parseBoolField(rawUseSSL, useSSL)
	}

	region := firstImageObjectEnv("CPA_IMAGE_R2_REGION", "CPA_R2_REGION", "R2_REGION", "AWS_REGION")
	if region == "" {
		region = "auto"
	}
	prefix := strings.Trim(firstImageObjectEnv("CPA_IMAGE_R2_PREFIX", "CPA_R2_PREFIX", "R2_IMAGE_PREFIX"), "/")
	if prefix == "" {
		prefix = "images"
	}

	return imageObjectStorageConfig{
		Endpoint:   endpoint,
		Bucket:     bucket,
		AccessKey:  accessKey,
		SecretKey:  secretKey,
		Region:     region,
		Prefix:     prefix,
		PublicBase: strings.TrimRight(firstImageObjectEnv("CPA_IMAGE_R2_PUBLIC_BASE_URL", "CPA_R2_PUBLIC_BASE_URL", "R2_PUBLIC_BASE_URL"), "/"),
		UseSSL:     useSSL,
	}, nil
}

func firstImageObjectEnv(names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	return ""
}

func normalizeImageObjectEndpoint(rawEndpoint string) (string, bool, error) {
	rawEndpoint = strings.TrimRight(strings.TrimSpace(rawEndpoint), "/")
	if rawEndpoint == "" {
		return "", false, errImageObjectStorageNotConfigured
	}
	if strings.HasPrefix(rawEndpoint, "http://") || strings.HasPrefix(rawEndpoint, "https://") {
		parsed, err := url.Parse(rawEndpoint)
		if err != nil {
			return "", false, fmt.Errorf("image object storage: parse endpoint: %w", err)
		}
		if parsed.Host == "" {
			return "", false, fmt.Errorf("image object storage: endpoint host is empty")
		}
		return parsed.Host, parsed.Scheme == "https", nil
	}
	return rawEndpoint, true, nil
}

func normalizeImagesResponseFormat(responseFormat string) string {
	responseFormat = strings.ToLower(strings.TrimSpace(responseFormat))
	if responseFormat != "" {
		return responseFormat
	}
	if override := strings.ToLower(strings.TrimSpace(os.Getenv("CPA_IMAGE_DEFAULT_RESPONSE_FORMAT"))); override != "" {
		return override
	}
	if imageObjectStorageConfigured() {
		return "url"
	}
	return "b64_json"
}

func imageCallResultToOpenAIItem(ctx context.Context, img imageCallResult, index int, responseFormat string) ([]byte, error) {
	item := []byte(`{}`)
	if responseFormat == "url" {
		if storage, err := getImageObjectStorage(); err == nil {
			stored, err := storage.putGeneratedImage(ctx, img, index)
			if err != nil {
				return nil, err
			}
			item, _ = sjson.SetBytes(item, "url", stored.URL)
			item, _ = sjson.SetBytes(item, "object_key", stored.Key)
		} else if errors.Is(err, errImageObjectStorageNotConfigured) {
			mt := mimeTypeFromOutputFormat(img.OutputFormat)
			item, _ = sjson.SetBytes(item, "url", "data:"+mt+";base64,"+img.Result)
		} else {
			return nil, err
		}
	} else {
		item, _ = sjson.SetBytes(item, "b64_json", img.Result)
	}
	if img.RevisedPrompt != "" {
		item, _ = sjson.SetBytes(item, "revised_prompt", img.RevisedPrompt)
	}
	return item, nil
}

func (s *imageObjectStorage) putGeneratedImage(ctx context.Context, img imageCallResult, index int) (*storedImageObject, error) {
	if s == nil || s.client == nil {
		return nil, errImageObjectStorageNotConfigured
	}
	data, err := decodeGeneratedImageBase64(img.Result)
	if err != nil {
		return nil, fmt.Errorf("image object storage: decode image %d: %w", index+1, err)
	}
	contentType := mimeTypeFromOutputFormat(img.OutputFormat)
	if contentType == "" || !strings.HasPrefix(contentType, "image/") {
		contentType = http.DetectContentType(data)
	}
	key := s.generatedImageObjectKey(contentType)
	_, err = s.client.PutObject(ctx, s.cfg.Bucket, key, bytes.NewReader(data), int64(len(data)), minio.PutObjectOptions{
		ContentType:  contentType,
		CacheControl: "public, max-age=31536000, immutable",
	})
	if err != nil {
		return nil, fmt.Errorf("image object storage: upload %s: %w", key, err)
	}
	objectURL, err := s.objectURL(ctx, key)
	if err != nil {
		return nil, err
	}
	return &storedImageObject{Key: key, URL: objectURL}, nil
}

func (s *imageObjectStorage) generatedImageObjectKey(contentType string) string {
	ext := imageExtensionFromContentType(contentType)
	name := uuid.NewString() + ext
	datePath := time.Now().UTC().Format("2006/01/02")
	if s.cfg.Prefix == "" {
		return datePath + "/" + name
	}
	return strings.Trim(s.cfg.Prefix, "/") + "/" + datePath + "/" + name
}

func (s *imageObjectStorage) objectURL(ctx context.Context, key string) (string, error) {
	if s.cfg.PublicBase != "" {
		escaped := strings.ReplaceAll(url.PathEscape(key), "%2F", "/")
		return s.cfg.PublicBase + "/" + escaped, nil
	}
	presigned, err := s.client.PresignedGetObject(ctx, s.cfg.Bucket, key, defaultImageObjectURLTTL, nil)
	if err != nil {
		return "", fmt.Errorf("image object storage: sign %s: %w", key, err)
	}
	return presigned.String(), nil
}

func decodeGeneratedImageBase64(payload string) ([]byte, error) {
	normalized := strings.Map(func(r rune) rune {
		if r == '\r' || r == '\n' || r == '\t' || r == ' ' {
			return -1
		}
		return r
	}, payload)
	return base64.StdEncoding.DecodeString(normalized)
}

func imageExtensionFromContentType(contentType string) string {
	switch strings.ToLower(strings.TrimSpace(strings.Split(contentType, ";")[0])) {
	case "image/jpeg", "image/jpg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	case "image/gif":
		return ".gif"
	case "image/png":
		return ".png"
	default:
		if ext := filepath.Ext(contentType); ext != "" {
			return ext
		}
		return ".png"
	}
}

func resetImageObjectStorageForTest() {
	imageObjectStorageOnce = sync.Once{}
	imageObjectStorageInst = nil
	imageObjectStorageErr = nil
}
