from impala_scope.analytics import detect_request_type, provider_from_host


def test_detects_major_provider_shapes() -> None:
    assert detect_request_type("/v1/chat/completions", "api.openai.com") == "openai.chat"
    assert detect_request_type("/v1/messages", "api.anthropic.com") == "anthropic.messages"
    assert detect_request_type("/model/x/converse-stream", "bedrock.us-east-1.amazonaws.com") == "aws.bedrock.converse"
    assert (
        detect_request_type("/v1/models/gemini:generateContent", "aiplatform.googleapis.com") == "google.vertex.gemini"
    )
    assert detect_request_type("/v1/embeddings", "api.groq.com") == "openai.embeddings"


def test_detects_new_inference_provider_but_not_health() -> None:
    body = {"model": "new", "prompt": "hi"}
    assert detect_request_type("/v2/inference/generate", "new.example", body) == "generic.inference"
    assert detect_request_type("/health", "new.example", body) is None
    assert provider_from_host("inference.acme.internal", "generic.inference") == "inference.acme.internal"


def test_detects_realtime_huggingface_and_sagemaker() -> None:
    assert detect_request_type("/v1/realtime", "api.openai.com") == "openai.realtime"
    assert detect_request_type("/models/acme/model", "api-inference.huggingface.co") == "huggingface.inference"
    assert (
        detect_request_type("/endpoints/model/invocations", "runtime.sagemaker.us-east-1.amazonaws.com")
        == "aws.sagemaker"
    )
    assert provider_from_host("router.huggingface.co", "huggingface.inference") == "huggingface"
    assert provider_from_host("runtime.sagemaker.us-east-1.amazonaws.com", "aws.sagemaker") == "aws-sagemaker"
