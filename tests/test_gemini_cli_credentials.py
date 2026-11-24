import pytest

from gfjproxy.providers.gemini_cli import Credentials


def _valid_credentials(**updates):
    data = {
        "access_token": "access-token",
        "id_token": "header.eyJzdWIiOiJ1c2VyIn0.signature",
        "refresh_token": "refresh-token",
        "expiry_date": 1_900_000_000_000,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "token_type": "Bearer",
    }
    data.update(updates)
    return data


def test_credentials_parse_accepts_valid_oauth_payload():
    credentials = Credentials.parse(_valid_credentials())
    assert credentials.access_token == "access-token"
    assert credentials.refresh_token == "refresh-token"


@pytest.mark.parametrize(
    "expiry_date",
    [True, "1900000000000", float("nan"), float("inf"), -1, 4_102_444_800_001],
)
def test_credentials_parse_rejects_invalid_expiry_values(expiry_date):
    with pytest.raises(TypeError, match="expiry_date"):
        Credentials.parse(_valid_credentials(expiry_date=expiry_date))


def test_credentials_parse_rejects_oversized_tokens():
    with pytest.raises(TypeError, match="access_token"):
        Credentials.parse(_valid_credentials(access_token="x" * 16_385))
