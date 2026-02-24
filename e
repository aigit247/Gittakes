from urllib.parse import urlparse, unquote

def normalize_s3_bucket_key(bucket: str, key: str) -> Tuple[str, str]:
    """
    Accepts:
      - bucket + key
      - OR bucket="" and key as 's3://bucket/key...'
    Returns sanitized (bucket, key) for boto3 get_object.
    """
    bucket = (bucket or "").strip()
    key = (key or "").strip()

    # If key is actually a full s3 uri
    if key.lower().startswith("s3://"):
        u = urlparse(key)
        bucket = (u.netloc or "").strip()
        key = (u.path or "").lstrip("/")

    # boto3 Key must not start with '/'
    key = key.lstrip("/")

    # If you built keys using URLs, decode %xx (esp %20)
    key = unquote(key)

    return bucket, key
